#!/usr/bin/env python3
"""
llm_diagnosis_new.py - ask an LLM to diagnose SMPL-X measurement errors for ONE subject.

TOLERANCE IS DECIDED BY PYTHON, NOT BY THE MODEL (pre-filtering):
  * Measurements with |error| <= tolerance are locked as WITHIN_TOLERANCE before the
    prompt is built. The model is told not to diagnose them.
  * Only out-of-tolerance measurements are sent for diagnosis.
  * If the model still answers for a locked measurement, that answer is discarded
    and logged in "overridden_by_rule".
  * If every measurement is within tolerance, no API call is made at all.

API keys are read from environment variables (never write them in this file):
  Windows cmd:   set ZHIPUAI_API_KEY=your_key
                 set DEEPSEEK_API_KEY=your_key

Examples:
  python llm_diagnosis.py --results results_S03updated2.json --subject S03 ^
      --meta subjects_meta.json --provider zhipu --model glm-4v-plus --image images\\S03_Front.jpg

  python llm_diagnosis.py --results results_S03updated2.json --subject S03 ^
      --meta subjects_meta.json --provider deepseek --model deepseek-chat --include-appearance

Add --dry-run to print the prompt without calling any API.
To run all subjects with all models, use run_all_new.py.
"""
import argparse
import base64
import json
import os
import sys
from datetime import datetime
from pathlib import Path

MEASUREMENTS = ["shoulder", "chest", "waist", "hip"]
IMPLAUSIBLE_PCT = 50.0  # errors above this are flagged as physically suspicious

# Categories the MODEL may choose from. WITHIN_TOLERANCE is deliberately NOT here:
# only the pipeline can assign it.
DIAGNOSIS_CATEGORIES = {
    "CLOTHING_VOLUME": "Garment adds volume or hangs away from the body, so the fitted surface is larger than the body. Can only explain an OVERestimate.",
    "DEPTH_AMBIGUITY": "One frontal photo does not show front-to-back body depth, so circumferences are poorly constrained. Most often an underestimate.",
    "POSE_OR_KEYPOINT": "Arm/leg pose or 2D keypoint placement distorted the fitted body, especially shoulder and chest.",
    "SCALE_OR_RESOLUTION": "Image resolution or the fixed camera/focal-length assumption distorted overall size or proportions.",
    "MODEL_MISMATCH": "The body model used for fitting (for example its gender) does not match the subject.",
    "LANDMARK_DEFINITION": "The measurement level or method differs from how the tape-measure ground truth was taken.",
    "FITTING_FAILURE": "The fitted mesh itself is implausible (a value far outside normal human proportions), so the optimisation failed rather than a specific image cause.",
    "GT_SUSPECT": "The ground-truth value looks implausible or possibly mis-recorded.",
    "UNDETERMINED": "The information provided does not support a specific cause.",
}
LOCKED = "WITHIN_TOLERANCE"
PROMPT_VERSION = "v4"   # v4: + code blocks CLOTHING_VOLUME for too-small values, + code size check


def model_mismatch_allowed(meta):
    """Code decides this fact: a body-model mismatch is only possible when the person's gender
    is known AND differs from the gender of the body model used."""
    if not meta:
        return False
    g = str(meta.get("subject_gender", "")).strip().lower()
    mg = str(meta.get("model_gender", "")).strip().lower()
    return g in ("male", "female") and mg in ("male", "female") and g != mg


ALL_CATEGORIES = {**DIAGNOSIS_CATEGORIES,
                  LOCKED: "Error is within tolerance; assigned by the pipeline, not the model."}

# Pipeline descriptions given to the model. "old" = first version of the measuring script.
PIPELINE_NOTES_NEW = """\
- Fitting: SMPLify-X on one frontal photo, body-only 2D keypoints from MediaPipe, CPU, one fixed focal length.
- Measuring: the fitted body shape is put into a standard standing pose (arms out) before measuring,
  so the pose in the photo does not change the numbers.
- Chest at nipple height, waist at belly-button height, hip at the widest part between crotch and
  belly button; each is a tape-like loop around the body. Shoulder: top of one shoulder bone to the
  other, over the back.
- Ground truth: tape measurements taken by the researcher (hip at the widest part)."""

PIPELINE_NOTES_HBW = """\
- Public dataset photo (HBW). Fitting: SMPLify-X on this one photo, 2D keypoints supplied with the
  dataset (OpenPose), gender-neutral body model, one fixed focal length.
- Measuring: the fitted body shape is put into a standard standing pose (arms out) before measuring,
  so the pose in the photo does not change the numbers.
- Chest at nipple height, waist at belly-button height, hip at the widest part between crotch and
  belly button. Shoulder: top of one shoulder bone to the other, over the back.
- Ground truth: the same measurements taken on the person's TRUE body (a 3D body scan), not a tape."""

PIPELINE_NOTES = """\
- Fitting: SMPLify-X on one frontal photo, body-only 2D keypoints from MediaPipe, CPU, one fixed focal length.
- Shoulder width: measured directly between SMPL-X shoulder vertices.
- Chest circumference: mesh cross-section with an ellipse-based depth correction.
- Waist and hip circumference: mesh cross-sections at fixed height ratios.
- Every photo was resized to the same width before keypoint extraction."""


# ---------------------------------------------------------------- data

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def signed_errors(data):
    """Recompute signed errors from predicted/ground truth (stored error_pct is absolute)."""
    rows = []
    pred, gt = data["predicted"], data["ground_truth"]
    for m in MEASUREMENTS:
        key = f"{m}_cm"
        if key not in pred or key not in gt:
            continue
        p, g = float(pred[key]), float(gt[key])
        diff = p - g
        if abs(diff) < 0.05:
            direction = "exact"
        else:
            direction = "overestimate" if diff > 0 else "underestimate"
        rows.append({
            "measurement": m,
            "predicted_cm": round(p, 1),
            "ground_truth_cm": round(g, 1),
            "signed_error_cm": round(diff, 1),
            "signed_error_pct": round(100.0 * diff / g, 1),
            "direction": direction,
        })
    return rows


def prefilter(rows, tol_pct):
    """THE TOLERANCE FIX: decide tolerance in code, before the model sees anything."""
    for r in rows:
        a = abs(r["signed_error_pct"])
        r["within_tolerance"] = a <= tol_pct
        r["implausible"] = a > IMPLAUSIBLE_PCT
    return rows


# ---------------------------------------------------------------- prompt

def build_prompt(subject, rows, meta, tol_pct, image_attached, include_appearance,
                 predicted_height=None, pipeline_notes=None, truth_name="tape-measure ground truth",
                 true_height=None, group_context=None):
    pipeline_notes = pipeline_notes or PIPELINE_NOTES
    to_diagnose = [r for r in rows if not r["within_tolerance"]]
    locked = [r for r in rows if r["within_tolerance"]]
    allowed = {k: v for k, v in DIAGNOSIS_CATEGORIES.items()
               if k != "MODEL_MISMATCH" or model_mismatch_allowed(meta)}
    cat_lines = "\n".join(f"- {k}: {v}" for k, v in allowed.items())
    if "MODEL_MISMATCH" not in allowed:
        cat_lines += ("\n(MODEL_MISMATCH is not available for this person: the body model's gender "
                      "matches the person, or the person's gender is not known.)")

    subject_lines = [f"Subject: {subject}"]
    if meta:
        if "subject_gender" in meta:
            subject_lines.append(f"Subject gender: {meta['subject_gender']}")
        if "model_gender" in meta:
            subject_lines.append(f"Body model gender used for fitting: {meta['model_gender']}")
        o, w = meta.get("original_width_px"), meta.get("fit_width_px")
        if o and w:
            note = " (photo was upscaled)" if o < w else ""
            subject_lines.append(f"Photo width before resizing: {o} px; width used for fitting: {w} px{note}")
        if meta.get("photo_setting"):
            subject_lines.append(f"Photo setting: {meta['photo_setting']}")
        if include_appearance and meta.get("clothing"):
            subject_lines.append(f"Clothing: {meta['clothing']}")
    if predicted_height and true_height:
        subject_lines.append(f"Height of the fitted mesh: {predicted_height} cm; true height: {true_height} cm")
        size_pct = 100.0 * (float(predicted_height) - float(true_height)) / float(true_height)
        if abs(size_pct) > 5:
            subject_lines.append(
                f"SIZE CHECK (computed by code): the fitted body is {abs(size_pct):.1f}% "
                f"{'taller' if size_pct > 0 else 'shorter'} than the true body, so the whole body was built at the "
                f"wrong overall size. This pushes measurements in the same direction ({'too large' if size_pct > 0 else 'too small'}); "
                f"consider SCALE_OR_RESOLUTION for errors in that direction.")
        else:
            subject_lines.append(
                f"SIZE CHECK (computed by code): the fitted body's height is within 5% of the true height "
                f"({size_pct:+.1f}%), so overall body size is not the main cause.")
    elif predicted_height:
        subject_lines.append(f"Height of the fitted mesh: {predicted_height} cm (true height not recorded)")

    def line(r):
        s = (f"- {r['measurement']}: predicted {r['predicted_cm']} cm, ground truth {r['ground_truth_cm']} cm, "
             f"error {r['signed_error_cm']:+} cm ({r['signed_error_pct']:+}%), {r['direction']}")
        if r["implausible"]:
            s += f"  [error above {IMPLAUSIBLE_PCT:.0f}%]"
        if not r["within_tolerance"] and r["direction"] == "underestimate":
            s += "  [too small, so CLOTHING_VOLUME is not possible here]"
        return s

    locked_block = "\n".join(line(r) for r in locked) or "- (none)"
    diag_block = "\n".join(line(r) for r in to_diagnose)
    example = to_diagnose[0]["measurement"] if to_diagnose else "waist"

    image_note = (
        "\nA photo of the subject is attached. Use it to judge clothing and pose, and say in the "
        "'evidence' field when a judgement comes from the photo.\n"
        if image_attached else ""
    )

    return f"""You are an expert in monocular 3D human body reconstruction (SMPL-X fitted with SMPLify-X) and anthropometric measurement. Diagnose why measurements taken from the fitted mesh disagree with the {truth_name}.

RULES
1. The pipeline has already checked the tolerance ({tol_pct:g}%). Measurements under ALREADY WITHIN TOLERANCE are finished: do NOT diagnose them and do NOT include them in your answer. You may use them as context (for example, an accurate hip next to an inaccurate waist).
2. Diagnose every measurement under NEEDS DIAGNOSIS, and only those.
3. The sign of the error matters. "overestimate" means predicted > ground truth; "underestimate" means predicted < ground truth. The cause you choose must be able to produce an error of that sign. Loose clothing adds volume, so it cannot explain an underestimate.
4. Use only the facts given below. Do not assume clothing, pose or camera details that are not stated. If the evidence does not support a specific cause, use UNDETERMINED.
4b. LOOK FIRST, THEN DECIDE. For each measurement, first fill "observation": describe what you actually see at that body level (in the photo if one is attached, otherwise in the clothing description): which garment covers it and whether it fits close to the body or hangs away from it. Only then choose the category, and it must agree with your observation. When more than one cause is possible, choose the one with the most direct evidence at that body level.
4c. Use the PATTERN ACROSS ALL SUBJECTS below. An error in the same direction as most other people points to a cause in the measuring method or the body model itself (for example LANDMARK_DEFINITION), not to this person's photo. An error that only this person has points to something in this person's photo.
5. Give exactly one category per measurement, chosen from:
{cat_lines}
6. For each diagnosis also write "explanation" and "fix" for a NON-EXPERT reader (for example a tailor, a shop customer or a physiotherapist). Use plain everyday English: no technical words such as SMPL-X, mesh, keypoints, optimisation, prior or depth ambiguity, and no category names. Mention the size of the error in cm. The "summary" must also be plain English.
   - If you blame clothing, name the exact garment that covers that body level and how it sits (for example "a thick winter coat adds bulk around the chest"). Only name a garment you can actually see in the photo or that is stated above; if a waistband or garment edge is hidden, say which garment covers it.
   - The "fix" must be one concrete action for next time: what to wear, how to stand, an extra photo to take, or a setting to change (for example "use the female body model"). Do not use vague words such as "consider", "adjust for" or "account for".
7. Reply with JSON only, no other text, in this shape:
{{"diagnoses": [{{"measurement": "{example}", "observation": "what is seen at this body level: garment and how it fits", "category": "CATEGORY", "confidence": "low|medium|high", "evidence": "one technical sentence naming the specific facts you used", "explanation": "one or two plain-English sentences saying what went wrong and why", "fix": "one short practical suggestion"}}], "summary": "at most two plain-English sentences about this person's results overall"}}

SUBJECT
{chr(10).join(subject_lines)}

PIPELINE
{pipeline_notes}

PATTERN ACROSS ALL SUBJECTS IN THIS STUDY
{group_context or "- (not available)"}

ALREADY WITHIN TOLERANCE (do not diagnose)
{locked_block}

NEEDS DIAGNOSIS
{diag_block}
{image_note}"""


# ---------------------------------------------------------------- API calls

def _key(name):
    k = os.environ.get(name)
    if not k:
        raise RuntimeError(f"Environment variable {name} is not set. In cmd run:  set {name}=your_key")
    return k


def encode_image(path, max_side=1600):
    """Send a JPEG of at most max_side pixels (large dataset photos can exceed API limits)."""
    try:
        import io
        from PIL import Image, ImageOps
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    except ImportError:
        suffix = Path(path).suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        with open(path, "rb") as f:
            return f"data:{mime};base64,{base64.b64encode(f.read()).decode('utf-8')}"


ZHIPU_THINKING = None   # None = model default; "enabled" or "disabled" (set by run_all_new.py --thinking)


def call_zhipu(model, prompt, image_path, temperature):
    # newer GLM models (e.g. glm-5.3-flash) need the new library: pip install zai-sdk
    try:
        from zai import ZhipuAiClient as Client
    except ImportError:
        from zhipuai import ZhipuAI as Client  # older library, fine for glm-4v-flash
    client = Client(api_key=_key("ZHIPUAI_API_KEY"))
    if image_path:
        content = [
            {"type": "image_url", "image_url": {"url": encode_image(image_path)}},
            {"type": "text", "text": prompt},
        ]
    else:
        content = prompt
    kwargs = {}
    if ZHIPU_THINKING in ("enabled", "disabled"):
        kwargs["thinking"] = {"type": ZHIPU_THINKING}
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=temperature,
        **kwargs,
    )
    msg = resp.choices[0].message
    return msg.content or ""


def call_deepseek(model, prompt, temperature):
    from openai import OpenAI  # pip install openai
    client = OpenAI(api_key=_key("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
    kwargs = {}
    if model == "deepseek-chat":
        kwargs["response_format"] = {"type": "json_object"}  # forces valid JSON
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        **kwargs,
    )
    return resp.choices[0].message.content


def call_model(provider, model, prompt, image_path, temperature):
    if provider == "zhipu":
        return call_zhipu(model, prompt, image_path, temperature)
    return call_deepseek(model, prompt, temperature)


# ---------------------------------------------------------------- parse / merge / check

def parse_response(text):
    if not text:
        return None
    i, j = text.find("{"), text.rfind("}")
    if i == -1 or j <= i:
        return None
    try:
        return json.loads(text[i:j + 1])
    except json.JSONDecodeError:
        return None


def merge_and_validate(parsed, rows, tol_pct, meta=None):
    """Combine rule-locked results with the model's diagnoses and check the model's answer.
    Returns (final_diagnoses, overridden_by_rule, problems)."""
    info = {r["measurement"]: r for r in rows}
    needs = [r for r in rows if not r["within_tolerance"]]
    problems, overridden, llm = [], [], {}

    diagnoses = parsed.get("diagnoses") if isinstance(parsed, dict) else None
    if needs and not isinstance(diagnoses, list):
        problems.append("response was not the expected JSON")
        diagnoses = []

    for d in diagnoses or []:
        if not isinstance(d, dict):
            continue
        m = str(d.get("measurement", "")).strip().lower()
        cat = d.get("category")
        if m not in info:
            problems.append(f"unknown measurement {m!r} in response")
            continue
        r = info[m]
        if r["within_tolerance"]:
            overridden.append({"measurement": m, "model_category": cat})
            continue
        if m in llm:
            problems.append(f"{m}: listed more than once")
            continue
        if cat == LOCKED:
            problems.append(f"{m}: model said WITHIN_TOLERANCE but error is {r['signed_error_pct']:+}%")
        elif cat not in DIAGNOSIS_CATEGORIES:
            problems.append(f"{m}: unknown category {cat!r}")
        if cat == "MODEL_MISMATCH" and not model_mismatch_allowed(meta):
            problems.append(f"{m}: MODEL_MISMATCH given, but the person's gender matches the model or is unknown")
        if not str(d.get("observation", "")).strip():
            problems.append(f"{m}: no observation written before the decision")
        if cat == "CLOTHING_VOLUME" and r["direction"] == "underestimate":
            problems.append(f"{m}: CLOTHING_VOLUME given for an underestimate")
        if r["implausible"] and cat not in ("FITTING_FAILURE", "GT_SUSPECT"):
            problems.append(f"{m}: error {r['signed_error_pct']:+}% is implausible but category is {cat}")
        llm[m] = d

    final = []
    for r in rows:
        m = r["measurement"]
        base = {"measurement": m, "signed_error_pct": r["signed_error_pct"], "direction": r["direction"]}
        if r["within_tolerance"]:
            final.append({**base, "category": LOCKED, "confidence": "rule", "source": "rule",
                          "evidence": f"|error| {abs(r['signed_error_pct'])}% <= {tol_pct:g}% tolerance",
                          "explanation": f"Within {tol_pct:g}% of the tape measure, so this measurement can be trusted.",
                          "fix": ""})
        elif m in llm:
            d = llm[m]
            if not str(d.get("explanation", "")).strip():
                problems.append(f"{m}: no plain-English explanation")
            final.append({**base, "category": d.get("category"), "confidence": d.get("confidence"),
                          "source": "llm", "evidence": d.get("evidence", ""),
                          "observation": d.get("observation", ""),
                          "explanation": d.get("explanation", ""), "fix": d.get("fix", "")})
        else:
            problems.append(f"{m}: missing from response")
            final.append({**base, "category": "MISSING", "confidence": None, "source": "missing",
                          "evidence": "", "explanation": "The model gave no answer for this measurement.", "fix": ""})
    return final, overridden, problems


# ---------------------------------------------------------------- one full diagnosis

def condition_name(image, include_appearance):
    return "image" if image else ("text+appearance" if include_appearance else "text")


def diagnose(results_path, subject, meta, provider, model, image=None,
             include_appearance=False, tol_pct=10.0, temperature=0.2, retries=1, dry_run=False,
             pipeline_notes=None, truth_name="tape-measure ground truth", truth_short="the tape measure",
             group_context=None):
    """Run the whole diagnosis for one subject and return a record (dict)."""
    data = load_json(results_path)
    rows = prefilter(signed_errors(data), tol_pct)
    prompt = build_prompt(subject, rows, meta, tol_pct,
                          image_attached=bool(image),
                          include_appearance=include_appearance,
                          predicted_height=data.get("predicted", {}).get("height_cm"),
                          pipeline_notes=pipeline_notes, truth_name=truth_name,
                          true_height=data.get("ground_truth", {}).get("height_cm"),
                          group_context=group_context)
    needs = [r for r in rows if not r["within_tolerance"]]

    record = {
        "subject": subject,
        "provider": provider,
        "model": model,
        "condition": condition_name(image, include_appearance),
        "temperature": temperature,
        "tolerance_pct": tol_pct,
        "prompt_version": PROMPT_VERSION,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "errors": rows,
        "prompt": prompt,
        "llm_called": False,
        "attempts": 0,
        "raw_response": None,
        "parsed": None,
        "api_error": None,
    }
    if dry_run:
        return record

    raw, parsed = None, None
    if needs:
        record["llm_called"] = True
        for attempt in range(1 + retries):
            record["attempts"] = attempt + 1
            raw, err = None, None
            for net_try in range(4):                 # connection problems: wait and try again
                try:
                    raw = call_model(provider, model, prompt, image, temperature)
                    err = None
                    break
                except Exception as e:  # network, auth, quota ...
                    err = f"{type(e).__name__}: {e}"
                    if not any(w in err for w in ("Connection", "Timeout", "timed out", "429")) or net_try == 3:
                        break
                    import time
                    wait = 15 * (net_try + 1)
                    print(f"    [retry] {err[:60]} ... waiting {wait}s")
                    time.sleep(wait)
            if err:
                record["api_error"] = err
                break
            parsed = parse_response(raw)
            if isinstance(parsed, dict) and isinstance(parsed.get("diagnoses"), list):
                record["api_error"] = None
                break

    final, overridden, problems = merge_and_validate(parsed, rows, tol_pct, meta)
    if record["api_error"]:
        problems.insert(0, f"API error: {record['api_error']}")

    if not needs:
        summary = (f"All four measurements are within {tol_pct:g}% of {truth_short}, "
                   "so this person's body measurements from the photo can be trusted.")
    else:
        summary = parsed.get("summary", "") if isinstance(parsed, dict) else ""

    record.update({
        "raw_response": raw,
        "parsed": parsed,
        "final_diagnoses": final,
        "summary": summary,
        "overridden_by_rule": overridden,
        "validation_problems": problems,
    })
    return record


def print_record(rec):
    print(f"\n=== {rec['subject']} | {rec['provider']}:{rec['model']} | {rec['condition']} ===")
    for d in rec["final_diagnoses"]:
        print(f"  {d['measurement']:9s} {d['signed_error_pct']:+6.1f}%  {str(d['category']):18s} "
              f"({d['confidence']}) {d['evidence']}")
        if d["source"] == "llm":
            print(f"            -> {d['explanation']}  Fix: {d['fix']}")
    print(f"\n  Summary: {rec['summary']}")
    if rec["overridden_by_rule"]:
        print("  [rule] ignored model answers for within-tolerance items: " +
              ", ".join(f"{o['measurement']}={o['model_category']}" for o in rec["overridden_by_rule"]))
    if rec["validation_problems"]:
        print("  [check] " + "; ".join(rec["validation_problems"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results JSON from extract_measurements.py")
    ap.add_argument("--subject", required=True, help="e.g. S03")
    ap.add_argument("--meta", help="subjects_meta.json")
    ap.add_argument("--provider", choices=["zhipu", "deepseek"], required=True)
    ap.add_argument("--model", required=True, help="model name as shown in the provider's console")
    ap.add_argument("--image", help="photo to send (zhipu vision models only)")
    ap.add_argument("--include-appearance", action="store_true",
                    help="also give the clothing text from the meta file")
    ap.add_argument("--tol-pct", type=float, default=10.0)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--retries", type=int, default=1, help="extra attempts if the reply is not valid JSON")
    ap.add_argument("--out")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.image and args.provider != "zhipu":
        sys.exit("--image is only supported with --provider zhipu (DeepSeek's chat API is text-only).")

    meta = None
    if args.meta:
        meta = load_json(args.meta).get(args.subject)
        if meta is None:
            print(f"[warn] no entry for {args.subject} in {args.meta}")

    rec = diagnose(args.results, args.subject, meta, args.provider, args.model,
                   image=args.image, include_appearance=args.include_appearance,
                   tol_pct=args.tol_pct, temperature=args.temperature,
                   retries=args.retries, dry_run=args.dry_run)

    if args.dry_run:
        print(rec["prompt"])
        return

    cond = rec["condition"].replace("+", "-")
    out_path = args.out or f"diagnosis_{args.subject}_{args.provider}_{cond}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    print_record(rec)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
