#!/usr/bin/env python3
"""
run_all_new.py - run every subject through every model and build a side-by-side comparison.

Default conditions (each is one column in the comparison table):
  GLM   | image            -> GLM vision model sees the photo (no clothing text)
  GLM   | text+appearance  -> GLM gets the clothing text instead of the photo
  DS    | text+appearance  -> DeepSeek gets the clothing text (it cannot see images)

A model is skipped automatically if its API key is not set, so you can run GLM
today and add DeepSeek later.

Windows cmd:
  set ZHIPUAI_API_KEY=your_key
  set DEEPSEEK_API_KEY=your_key
  python run_all_new.py --glm-model glm-4v-plus

  python run_all_new.py --dry-run          (check everything without calling any API)
  python run_all_new.py --repeats 3        (consistency test: each call made 3 times)

Outputs (in --out-dir, default "diagnoses"):
  <subject>_<provider>_<condition>_r<k>.json   one file per call
  comparison.csv                               one row per subject x measurement
  comparison.md                                same table in Markdown (paste into thesis/PPT)
  model_summary.csv                            validation problems + consistency per condition
  report.md                                    plain-English report for every subject (open in PyCharm)
"""
import argparse
import csv
import json
import os
import time
from collections import Counter
from pathlib import Path

import llm_diagnosis_new as L

# Which results to diagnose (--dataset):
#   own  = my 6 people, measured with measure_body.py (shape_results\), hip = widest hip
#   hbw  = the 80 HBW photos, measured with measure_body.py (hbw_pred_measurements_v2.json)
#   old  = my 6 people with the first measuring script (the original results_S0x_updated2.json)
SHAPE_FILES = {
    "S01": "shape_results/results_S01_shape.json",
    "S02": "shape_results/results_S02_shape.json",
    "S03": "shape_results/results_S03_shape_male.json",
    "S05": "shape_results/results_S05_shape.json",
    "S06": "shape_results/results_S06_shape.json",
    "S07": "shape_results/results_S07_shape.json",
}
FOUR = ["shoulder_cm", "chest_cm", "waist_cm", "hip_cm"]
RESULT_FILES = {
    "S01": "results_S01updated2.json",
    "S02": "results_S02updated2.json",
    "S03": "results_S03updated2.json",
    "S05": "results_S05updated2.json",
    "S06": "results_S06updated2.json",
    "S07": "results_S07updated2.json",
}
KEY_ENV = {"zhipu": "ZHIPUAI_API_KEY", "deepseek": "DEEPSEEK_API_KEY"}
SHORT = {"zhipu": "GLM", "deepseek": "DS"}


NAMES = {"shoulder": "Shoulder width", "chest": "Chest", "waist": "Waist", "hip": "Hip"}


def write_report(records, labels, subjects, meta_all, tol_pct, path, truth_word="Tape measure"):
    """Plain-English report. Uses the first condition (photo if available) as the main answer."""
    L_ = ["# Body measurement report", "",
          f"Each measurement from the photo is compared with: {truth_word.lower()}. "
          f"Differences within {tol_pct:g}% are treated as reliable.", ""]
    for subj in subjects:
        recs = {l: records[(subj, l)][0] for l in labels if (subj, l) in records}
        if not recs:
            continue
        main_label = next(iter(recs))
        main = recs[main_label]
        meta = meta_all.get(subj, {})
        desc = ", ".join(x for x in [meta.get("subject_gender", ""), meta.get("clothing", "")]
                         if x and x != "unknown")
        L_ += [f"## {subj}" + (f" ({desc})" if desc else ""), "", f"**Overall:** {main['summary']}", "",
               f"| Measurement | From photo | {truth_word} | Difference | Verdict |",
               "|---|---|---|---|---|"]
        for e in main["errors"]:
            verdict = "reliable" if e["within_tolerance"] else "needs attention"
            L_.append(f"| {NAMES[e['measurement']]} | {e['predicted_cm']} cm | {e['ground_truth_cm']} cm | "
                      f"{e['signed_error_cm']:+} cm ({e['signed_error_pct']:+}%) | {verdict} |")
        L_.append("")
        for d in main["final_diagnoses"]:
            if d["source"] == "rule":
                continue
            e = next(x for x in main["errors"] if x["measurement"] == d["measurement"])
            word = "too large" if e["signed_error_cm"] > 0 else "too small"
            L_.append(f"**{NAMES[d['measurement']]} is {abs(e['signed_error_cm'])} cm {word} "
                      f"({e['signed_error_pct']:+}%).** {d['explanation']}")
            if d.get("fix"):
                L_.append(f"*Suggested fix:* {d['fix']}")
            if d.get("observation"):
                L_.append(f"*What the model saw:* {d['observation']}")
            others = [f"{l}: {next(x['category'] for x in r['final_diagnoses'] if x['measurement'] == d['measurement'])}"
                      for l, r in recs.items() if l != main_label]
            L_.append(f"<sub>Cause ({main_label}): {d['category']}, confidence {d['confidence']}"
                      + (f" | other runs: {'; '.join(others)}" if others else "") + "</sub>")
            L_.append("")
        L_.append("")
    Path(path).write_text("\n".join(L_), encoding="utf-8")


def build_inputs(dataset, args, meta_all, inputs_dir):
    """Return a list of (id, results_json_path, meta, image_path). New-format results are
    rewritten into the simple predicted / ground_truth form the diagnosis expects
    (with hip = widest hip), and saved in inputs_dir so every input can be checked."""
    items = []
    if dataset == "old":
        for sid, fn in RESULT_FILES.items():
            p = Path(args.results_dir) / fn
            if p.exists():
                items.append((sid, str(p), meta_all.get(sid), find_image(args.images_dir, sid)))
            else:
                print(f"[skip] {sid}: {p} not found")
        return items
    inputs_dir.mkdir(parents=True, exist_ok=True)

    def pick(d):
        out = {k: d[k] for k in FOUR if k in d}
        if "hip_widest_cm" in d:
            out["hip_cm"] = d["hip_widest_cm"]       # widest hip = how the tape was taken
        if "height_cm" in d:
            out["height_cm"] = d["height_cm"]
        return out

    if dataset == "own":
        for sid, fn in SHAPE_FILES.items():
            p = Path(args.results_dir) / fn
            if not p.exists():
                print(f"[skip] {sid}: {p} not found")
                continue
            d = L.load_json(p)
            gt = {k: d["ground_truth"][k] for k in FOUR + ["height_cm"] if k in d["ground_truth"]}
            norm = {"source": str(p), "predicted": pick(d["predicted"]), "ground_truth": gt}
            q = inputs_dir / f"{sid}.json"
            q.write_text(json.dumps(norm, indent=2), encoding="utf-8")
            items.append((sid, str(q), meta_all.get(sid), find_image(args.images_dir, sid)))
    else:  # hbw
        d = L.load_json(args.hbw_results)
        for name, ph in d["photos"].items():
            norm = {"source": args.hbw_results, "predicted": pick(ph["predicted"]),
                    "ground_truth": pick(ph["true"])}
            q = inputs_dir / f"{name}.json"
            q.write_text(json.dumps(norm, indent=2), encoding="utf-8")
            meta = {"model_gender": "neutral",
                    "photo_setting": "lab studio photo" if ph["setting"] == "lab"
                    else "everyday photo taken outside the lab"}
            img = None
            for ext in (".png", ".jpg", ".jpeg"):
                cand = Path(args.hbw_images) / f"{name}{ext}"
                if cand.exists():
                    img = str(cand)
                    break
            items.append((name, str(q), meta, img))
    return items


def pattern_across(items, tol_pct):
    """Plain summary of how each measurement goes wrong across ALL subjects (given to the LLM)."""
    errs = {m: [] for m in L.MEASUREMENTS}
    for _, path, _, _ in items:
        for r in L.signed_errors(L.load_json(path)):
            errs[r["measurement"]].append(r["signed_error_pct"])
    lines = []
    for m, e in errs.items():
        if not e:
            continue
        n = len(e)
        small = sum(x < -5 for x in e)
        big = sum(x > 5 for x in e)
        lines.append(f"- {m}: average {sum(e) / n:+.1f}%; more than 5% too small in {small} of {n}, "
                     f"more than 5% too big in {big} of {n}")
    return "\n".join(lines)


def find_image(images_dir, subject):
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        p = Path(images_dir) / f"{subject}_Front{ext}"
        if p.exists():
            return str(p)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["own", "hbw", "old"], default="own",
                    help="own = my 6 people (new measuring), hbw = 80 HBW photos, old = first measuring script")
    ap.add_argument("--hbw-results", default="hbw_pred_measurements_v2.json")
    ap.add_argument("--hbw-images", default=os.path.join("hbw_selected", "images"))
    ap.add_argument("--results-dir", default=".")
    ap.add_argument("--meta", default="subjects_meta.json")
    ap.add_argument("--images-dir", default="images", help="expects files like S03_Front.jpg")
    ap.add_argument("--glm-model", default="glm-4v-flash", help="GLM vision model name from your Zhipu console")
    ap.add_argument("--deepseek-model", default="deepseek-chat")
    ap.add_argument("--providers", nargs="+", default=["zhipu", "deepseek"], choices=["zhipu", "deepseek"])
    ap.add_argument("--subjects", nargs="+", help="only these ids (default: all)")
    ap.add_argument("--tol-pct", type=float, default=10.0)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--repeats", type=int, default=1, help="calls per condition (use 3+ for a consistency test)")
    ap.add_argument("--out-dir", default="diagnoses")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between API calls")
    ap.add_argument("--thinking", choices=["enabled", "disabled"],
                    help="GLM 'thinking' mode for newer models such as glm-5.3-flash (default: model's own)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    L.ZHIPU_THINKING = args.thinking
    meta_all = L.load_json(args.meta)
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    items = build_inputs(args.dataset, args, meta_all, out / "inputs")
    if args.subjects:
        items = [it for it in items if it[0] in args.subjects]
    args.subjects = [it[0] for it in items]
    notes = {"own": L.PIPELINE_NOTES_NEW, "hbw": L.PIPELINE_NOTES_HBW, "old": None}[args.dataset]
    truth = ("true body (3D scan)", "the true body (3D scan)", "True body") if args.dataset == "hbw" \
        else ("tape-measure ground truth", "the tape measure", "Tape measure")
    meta_all = {it[0]: (it[2] or {}) for it in items}
    group_context = pattern_across(items, args.tol_pct)
    print("Pattern across subjects given to the LLM:\n" + group_context + "\n")

    # ---- decide which conditions will run
    conditions = []  # (label, provider, model, use_image, include_appearance)
    for prov in args.providers:
        if not args.dry_run and not os.environ.get(KEY_ENV[prov]):
            print(f"[skip] {prov}: {KEY_ENV[prov]} is not set")
            continue
        if prov == "zhipu":
            conditions.append(("GLM|image", prov, args.glm_model, True, False))
            if args.dataset != "hbw":
                conditions.append(("GLM|text+appearance", prov, args.glm_model, False, True))
        elif args.dataset == "hbw":
            print("[skip] deepseek: HBW photos have no clothing description, so a text-only model has nothing to go on")
        else:
            conditions.append(("DS|text+appearance", prov, args.deepseek_model, False, True))
    if not conditions:
        raise SystemExit("No model to run. Set at least one API key (see top of this file).")

    # ---- run
    records = {}  # (subject, label) -> list of records
    for subj, res_path, meta, image in items:
        for label, prov, model, use_image, appearance in conditions:
            if use_image and not image:
                print(f"[skip] {subj} {label}: no photo found")
                continue
            for k in range(1, args.repeats + 1):
                rec = L.diagnose(str(res_path), subj, meta, prov, model,
                                 image=image if use_image else None,
                                 include_appearance=appearance,
                                 tol_pct=args.tol_pct, temperature=args.temperature,
                                 dry_run=args.dry_run, pipeline_notes=notes,
                                 truth_name=truth[0], truth_short=truth[1],
                                 group_context=group_context)
                if args.dry_run:
                    print(f"[dry-run] {subj} {label} r{k}: prompt OK ({len(rec['prompt'])} chars)")
                    continue
                fname = f"{subj}_{prov}_{rec['condition'].replace('+', '-')}_r{k}.json"
                with open(out / fname, "w", encoding="utf-8") as f:
                    json.dump(rec, f, indent=2, ensure_ascii=False)
                records.setdefault((subj, label), []).append(rec)
                cats = ", ".join(f"{d['measurement']}={d['category']}" for d in rec["final_diagnoses"]
                                 if d["source"] != "rule") or "all within tolerance"
                flag = f"  [{len(rec['validation_problems'])} problem(s)]" if rec["validation_problems"] else ""
                print(f"{subj} {label} r{k}: {cats}{flag}")
                if rec["llm_called"]:
                    time.sleep(args.pause)

    if args.dry_run:
        print("\nDry run finished. Nothing was sent to any API.")
        return

    labels = [c[0] for c in conditions]

    # ---- comparison table (one row per subject x measurement)
    rows = []
    for subj in args.subjects:
        any_rec = next((records[(subj, l)][0] for l in labels if (subj, l) in records), None)
        if not any_rec:
            continue
        for e in any_rec["errors"]:
            m = e["measurement"]
            row = {"subject": subj, "measurement": m,
                   "error_pct": f"{e['signed_error_pct']:+.1f}",
                   "status": "within" if e["within_tolerance"] else "DIAGNOSE"}
            for l in labels:
                recs = records.get((subj, l), [])
                cats = [next(d["category"] for d in r["final_diagnoses"] if d["measurement"] == m) for r in recs]
                if not cats:
                    row[l] = "-"
                    continue
                top, n = Counter(cats).most_common(1)[0]
                row[l] = top if len(cats) == 1 else f"{top} ({n}/{len(cats)})"
            rows.append(row)

    fields = ["subject", "measurement", "error_pct", "status"] + labels
    with open(out / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    md = ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    for r in rows:
        cells = [r[k] if r["status"] == "DIAGNOSE" or k in fields[:4] else "·" for k in fields]
        md.append("| " + " | ".join(cells) + " |")
    md.append("\n· = within tolerance (assigned by rule, not by the model)")
    (out / "comparison.md").write_text("\n".join(md), encoding="utf-8")

    # ---- per-condition summary: validation problems + consistency
    summ = []
    for l in labels:
        recs = [r for (s, lab), rs in records.items() if lab == l for r in rs if r["llm_called"]]
        diag_items = sum(1 for r in recs for d in r["final_diagnoses"] if d["source"] != "rule")
        problems = sum(len(r["validation_problems"]) for r in recs)
        overrides = sum(len(r["overridden_by_rule"]) for r in recs)
        # consistency: share of (subject, measurement) where all repeats agree
        agree = total = 0
        for (s, lab), rs in records.items():
            if lab != l or len(rs) < 2 or not rs[0]["llm_called"]:
                continue
            for d in rs[0]["final_diagnoses"]:
                if d["source"] == "rule":
                    continue
                cats = {next(x["category"] for x in r["final_diagnoses"] if x["measurement"] == d["measurement"])
                        for r in rs}
                total += 1
                agree += len(cats) == 1
        summ.append({"condition": l, "llm_calls": len(recs), "diagnosed_items": diag_items,
                     "validation_problems": problems, "ignored_within_tol_answers": overrides,
                     "consistency": f"{agree}/{total}" if total else "n/a (repeats=1)"})
    with open(out / "model_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)

    write_report(records, labels, args.subjects, meta_all, args.tol_pct, out / "report.md", truth[2])

    print("\n" + "\n".join(md))
    print("\nPer-condition summary:")
    for s in summ:
        print("  " + ", ".join(f"{k}={v}" for k, v in s.items()))
    print(f"\nAll files saved in {out}/")
    print(f"Plain-English report: {out / 'report.md'}  (open it in PyCharm and click Preview)")


if __name__ == "__main__":
    main()
