r"""
compare_scripts_hbw.py - run YOUR extract_measurements.py on HBW and compare it with measure_body.py.

It uses YOUR extract_measurements.py exactly as it is (imports it) on:
  - the 10 true HBW bodies   (D:\HBW\HBW\smplx\val\*.obj)
  - the 80 fitted HBW bodies (output_hbw\meshes\<photo>\000.obj)
For the TRUE bodies the chest is measured without the 1.45 boost, because the boost is a
correction for fitted bodies, not for a true body.

Measurements compared: height, shoulder, chest, waist, hip.

Run from the project folder:
  python compare_scripts_hbw.py --true D:\HBW\HBW\smplx\val --fitted output_hbw\meshes --new hbw_pred_measurements.json
It only reads files and writes compare_scripts_hbw.csv.
"""
import argparse
import contextlib
import csv
import glob
import io
import json
import os

import numpy as np

import extract_measurements as em          # YOUR script, unchanged

KEYS = ["height_cm", "shoulder_cm", "chest_cm", "waist_cm", "hip_cm"]


def quiet(fn, *a):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a)


def load(path):
    if path.lower().endswith(".npy"):
        return np.load(path).astype(np.float64)
    return quiet(em.load_obj, path)


def measure_true(v):
    out = {"height_cm": em.measure_height(v), "shoulder_cm": quiet(em.measure_shoulder, v)}
    out["chest_cm"] = quiet(em.measure_circumference, v, em.Y_RATIOS["chest"], "chest_true")  # no boost
    out["waist_cm"] = quiet(em.measure_circumference, v, em.Y_RATIOS["waist"], "waist")
    out["hip_cm"] = quiet(em.measure_circumference, v, em.Y_RATIOS["hip"], "hip")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--true", required=True, help=r"folder with the true HBW bodies, e.g. D:\HBW\HBW\smplx\val")
    ap.add_argument("--fitted", default=os.path.join("output_hbw", "meshes"))
    ap.add_argument("--new", default="hbw_pred_measurements.json", help="results of measure_body.py on HBW")
    ap.add_argument("--out", default="compare_scripts_hbw.csv")
    args = ap.parse_args()

    truth = {}
    for f in sorted(glob.glob(os.path.join(args.true, "*.obj")) or glob.glob(os.path.join(args.true, "*.npy"))):
        truth[os.path.splitext(os.path.basename(f))[0]] = measure_true(load(f))
    if not truth:
        raise SystemExit(f"No true bodies found in {args.true}")

    rows = []
    for d in sorted(os.listdir(args.fitted)):
        parts = d.split("_")
        mesh = os.path.join(args.fitted, d, "000.obj")
        if len(parts) < 3 or parts[0] not in truth or parts[1] not in ("lab", "wild") or not os.path.exists(mesh):
            continue
        pred = quiet(em.extract_all, load(mesh))
        gt = truth[parts[0]]
        rows.append({"photo": d, "subject": parts[0], "setting": parts[1],
                     **{f"pred_{k}": pred[k] for k in KEYS}, **{f"true_{k}": gt[k] for k in KEYS}})
    if not rows:
        raise SystemExit(f"No fitted HBW bodies found in {args.fitted}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    new = json.load(open(args.new, encoding="utf-8"))["summary"] if os.path.exists(args.new) else None

    def stats(sel, k):
        e = np.array([abs(r[f"pred_{k}"] - r[f"true_{k}"]) / r[f"true_{k}"] for r in sel])
        return 100 * np.nanmean(e), int(np.sum(e <= 0.10))

    print(f"\nHBW, {len(rows)} photos of {len(truth)} people: YOUR script vs NEW script (measure_body.py)")
    print("=" * 70)
    print(f"{'Group':<6}{'Measurement':<12}{'Your script':>17}{'New script':>17}")
    print(f"{'':<18}{'error  /  within 10%':>34}")
    print("-" * 70)
    for g in ("all", "lab", "wild"):
        sel = [r for r in rows if g == "all" or r["setting"] == g]
        for k in KEYS:
            m, n = stats(sel, k)
            nn = ""
            if new and g in new:
                key = "hip_widest_cm" if (k == "hip_cm" and "hip_widest_cm" in new[g]) else k
                x = new[g].get(key)
                if x:
                    nn = f"{x['mean_abs_error_pct']:5.1f}%  {x['within_10pct']:>6}"
            print(f"{g:<6}{k[:-3]:<12}{m:>9.1f}%  {n:>3}/{len(sel):<3}{nn:>17}")
        print("-" * 70)
    print(f"[ok] details saved -> {args.out}")


if __name__ == "__main__":
    main()
