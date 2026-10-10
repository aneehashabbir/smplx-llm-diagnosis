r"""
measure_body.py - ONE measuring script for your own subjects AND for HBW.

How it measures (same method as the SHAPY paper / HBW benchmark):
  1. Every body is put in the same standard pose (standing, arms straight out:
     the "T-pose"), so pose cannot change the numbers.
  2. Chest, waist and hip are measured around the body at fixed BODY LANDMARKS
     (not at fixed heights):
        chest = at nipple height        waist = at belly-button height
        hip   = at crotch height        height = top of head to heel
        hip_widest = the widest girth between the crotch and halfway up to the
                     belly button (= the widest part of hips/buttocks, like a tape
                     measure; this is how my own subjects were measured)
     Each one is the length of a tape wrapped around that horizontal cut.
  3. Shoulder = from the top of one shoulder bone to the other, over the back,
     like a taut tape (the method we checked on S03).

It NEVER overwrites a file that already exists, so your old results are safe.

------------------------------------------------------------------------------
YOUR OWN SUBJECTS (uses the SMPLify-X result file .pkl, not the .obj)
  python measure_body.py --pkl output\results\S03_Front\000.pkl --gender male ^
      --subject_id S03 --gt ground_truth\measurements.csv --out results_S03_shape.json

HBW TRUE BODIES (all 10 at once)
  python measure_body.py --hbw_gt_folder D:\HBW\HBW_low_resolution\smplx\val --out hbw_gt_measurements.json

HBW PHOTOS FITTED BY SMPLify-X, compared with the true bodies (all at once)
  python measure_body.py --hbw_pred_folder results --hbw_gt hbw_gt_measurements.json ^
      --gender neutral --out hbw_pred_measurements.json

ONE BODY ALREADY IN T-POSE (.npy or .obj with 10475 vertices)
  python measure_body.py --tpose D:\HBW\HBW_low_resolution\smplx\val\012.npy --subject_id HBW012 --out test_012.json
------------------------------------------------------------------------------
"""
import argparse
import csv
import glob
import json
import os
import pickle
import sys

import numpy as np
from scipy.spatial import ConvexHull

N_VERTS = 10475

# Landmark points on the SMPL-X surface (triangle index + position inside it),
# taken from SHAPY's smplx_measurements.yaml so the measurements match HBW.
LANDMARKS = {
    "HeadTop":     {"face_idx": 2581,  "bc": [0.8277337276382795, 0.1422200962169292, 0.030046176144791284]},
    "HeelLeft":    {"face_idx": 15605, "bc": [0.0, 1.0, 0.0]},
    "NippleRight": {"face_idx": 18402, "bc": [0.0, 0.0, 1.0]},
    "BellyButton": {"face_idx": 19229, "bc": [0.0, 1.0, 0.0]},
    "Crotch":      {"face_idx": 6194,  "bc": [0.0, 0.0, 1.0]},
}
CUTS = {"chest_cm": "NippleRight", "waist_cm": "BellyButton", "hip_cm": "Crotch"}
LEFT_SHOULDER_JOINT, RIGHT_SHOULDER_JOINT = 16, 17


# ----------------------------------------------------------------------------- loading
def load_model(models_dir, gender):
    path = os.path.join(models_dir, f"SMPLX_{gender.upper()}.npz")
    if not os.path.exists(path):
        sys.exit(f"Model file not found: {path}")
    m = np.load(path, allow_pickle=True)

    def dense(a):
        if a.dtype == object:
            a = a.item()
            a = a.toarray() if hasattr(a, "toarray") else np.asarray(a)
        return np.asarray(a)

    return {
        "v_template": dense(m["v_template"]).astype(np.float64),
        "shapedirs": dense(m["shapedirs"]).astype(np.float64),
        "faces": dense(m["f"]).astype(np.int64),
        "J_regressor": dense(m["J_regressor"]).astype(np.float64),
    }


def load_vertices(path):
    if path.lower().endswith(".npy"):
        v = np.load(path).astype(np.float64)
    else:
        v = np.array([[float(x) for x in line.split()[1:4]]
                      for line in open(path, encoding="utf-8", errors="ignore")
                      if line.startswith("v ")], dtype=np.float64)
    if v.shape != (N_VERTS, 3):
        sys.exit(f"{path}: expected an SMPL-X body with {N_VERTS} points, got {v.shape}.")
    return v


def tpose_from_pkl(pkl_path, model):
    """Body shape from SMPLify-X, put into the standard T-pose (no pose, no face expression)."""
    with open(pkl_path, "rb") as f:
        res = pickle.load(f, encoding="latin1")
    betas = np.asarray(res["betas"], dtype=np.float64).reshape(-1)
    n = min(len(betas), model["shapedirs"].shape[2])
    return model["v_template"] + model["shapedirs"][:, :, :n] @ betas[:n], betas


# ----------------------------------------------------------------------------- geometry
def landmark(v, faces, name):
    lm = LANDMARKS[name]
    tri = v[faces[lm["face_idx"]]]
    return (tri * np.asarray(lm["bc"])[:, None]).sum(axis=0)


def tape_around(v, faces, height):
    """Length of a tape wrapped around the horizontal cut through the body at this height."""
    tris = v[faces]                                   # (F, 3, 3)
    d = tris[:, :, 1] - height                        # signed distance of each corner to the cut
    pts = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        cross = (d[:, a] * d[:, b]) < 0
        if not np.any(cross):
            continue
        t = d[cross, a] / (d[cross, a] - d[cross, b])
        p = tris[cross, a] + t[:, None] * (tris[cross, b] - tris[cross, a])
        pts.append(p[:, [0, 2]])
    if not pts:
        return float("nan")
    pts = np.vstack(pts)
    hull = ConvexHull(pts)
    ring = pts[hull.vertices]
    ring = np.vstack([ring, ring[:1]])
    return float(np.sum(np.linalg.norm(np.diff(ring, axis=0), axis=1)))


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def shoulder_over_back(v, J_regressor):
    """Top of one shoulder bone to the other, over the back, like a taut tape (T-pose, front = +z)."""
    J = J_regressor @ v
    tops = []
    for j in (J[LEFT_SHOULDER_JOINT], J[RIGHT_SHOULDER_JOINT]):
        side = np.sign(j[0])
        dx = (v[:, 0] - j[0]) * side
        near = (dx >= 0) & (dx < 0.03) & (np.abs(v[:, 2] - j[2]) < 0.06) \
               & (v[:, 1] > j[1]) & (v[:, 1] < j[1] + 0.12)
        idx = np.where(near)[0]
        if len(idx) == 0:
            return float("nan"), None
        tops.append(v[idx[np.argmax(v[idx, 1])]])
    a, b = tops
    y = (a[1] + b[1]) / 2
    band = v[np.abs(v[:, 1] - y) < 0.015]
    lo, hi = sorted([a[0], b[0]])
    pts = band[(band[:, 0] > lo) & (band[:, 0] < hi)][:, [0, 2]]
    pts = np.vstack([pts, [a[0], a[2]], [b[0], b[2]]])
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    chain = []                       # back side = low z; keep the outer (taut) edge
    for p in pts:
        while len(chain) >= 2 and _cross(chain[-2], chain[-1], p) <= 0:
            chain.pop()
        chain.append(p)
    chain = np.array(chain)
    straight = float(np.linalg.norm(a - b))
    return float(np.sum(np.linalg.norm(np.diff(chain, axis=0), axis=1))), straight


def measure(v, model):
    faces = model["faces"]
    out = {}
    top, heel = landmark(v, faces, "HeadTop"), landmark(v, faces, "HeelLeft")
    out["height_cm"] = round(abs(top[1] - heel[1]) * 100, 1)
    sh, straight = shoulder_over_back(v, model["J_regressor"])
    out["shoulder_cm"] = round(sh * 100, 1)
    for key, lm in CUTS.items():
        out[key] = round(tape_around(v, faces, landmark(v, faces, lm)[1]) * 100, 1)
    # widest hip: scan from the crotch up to halfway to the belly button, keep the largest girth
    y_crotch = landmark(v, faces, "Crotch")[1]
    y_navel = landmark(v, faces, "BellyButton")[1]
    ys = np.linspace(y_crotch + 0.005, y_crotch + 0.5 * (y_navel - y_crotch), 25)
    out["hip_widest_cm"] = round(max(tape_around(v, faces, y) for y in ys) * 100, 1)
    details = {"shoulder_straight_bone_to_bone_cm": round(straight * 100, 1) if straight else None}
    return out, details


# ----------------------------------------------------------------------------- tape values & errors
def load_gt(csv_path, subject_id):
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("subject_id", "").strip() == subject_id:
                gt = {}
                for k, val in row.items():
                    try:
                        gt[k.strip().lower().replace(" ", "_")] = float(val)
                    except (TypeError, ValueError):
                        pass
                return gt
    print(f"[!] {subject_id} not found in {csv_path}")
    return {}


def errors(pred, gt):
    """Compare with the tape. The tape hip was taken at the widest part, so it is compared
    with BOTH hip_cm (HBW standard, crotch level) and hip_widest_cm."""
    gt = dict(gt)
    if "hip_cm" in gt:
        gt["hip_widest_cm"] = gt["hip_cm"]
    out = {}
    for k, p in pred.items():
        if k in gt and gt[k] and not np.isnan(p):
            out[k] = {"predicted_cm": p, "ground_truth_cm": round(gt[k], 1),
                      "mae_cm": round(abs(p - gt[k]), 1),
                      "error_pct": round(abs(p - gt[k]) / gt[k] * 100, 1)}
    return out


def save(path, data, overwrite):
    if os.path.exists(path) and not overwrite:
        sys.exit(f"[stop] {path} already exists. Choose a new --out name "
                 f"(your old results are never overwritten).")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[ok] saved -> {path}")


METHOD = ("T-pose; chest/waist/hip = convex tape around horizontal cut at nipple/belly-button/crotch "
          "(SHAPY/HBW definition); shoulder = acromion to acromion over the back; height = head top to heel")


def print_table(pred, err):
    print("-" * 60)
    print(f"{'Measurement':<14}{'Predicted':>11}{'Tape':>9}{'Error':>10}")
    for k, p in pred.items():
        e = err.get(k)
        if e:
            sign = 1 if p >= e["ground_truth_cm"] else -1
            print(f"{k[:-3]:<14}{p:>9.1f} cm{e['ground_truth_cm']:>7.1f}{sign * e['error_pct']:>+9.1f}%")
        else:
            print(f"{k[:-3]:<14}{p:>9.1f} cm")
    print("-" * 60)
    if "hip_widest_cm" in err:
        print("(your tape hip was taken at the widest part -> compare with hip_widest)")


def measure_hbw_predictions(args, model):
    """Measure every fitted HBW photo (folders like 012_lab_00020) and compare with the true body."""
    if not args.hbw_gt or not os.path.exists(args.hbw_gt):
        sys.exit("Give the true-body file with --hbw_gt hbw_gt_measurements.json")
    if os.path.exists(args.out) and not args.overwrite:
        sys.exit(f"[stop] {args.out} already exists. Choose a new --out name.")
    truth = json.load(open(args.hbw_gt, encoding="utf-8"))["bodies"]
    keys = ["height_cm", "shoulder_cm", "chest_cm", "waist_cm", "hip_cm", "hip_widest_cm"]
    first = next(iter(truth.values()))
    if "hip_widest_cm" not in first:
        print("[!] the true-body file has no hip_widest_cm: re-measure the true bodies with this version first")
        keys.remove("hip_widest_cm")
    photos, rows = {}, []
    for d in sorted(os.listdir(args.hbw_pred_folder)):
        parts = d.split("_")
        pkl = os.path.join(args.hbw_pred_folder, d, "000.pkl")
        if len(parts) < 3 or parts[0] not in truth or parts[1] not in ("lab", "wild") or not os.path.exists(pkl):
            continue
        v, betas = tpose_from_pkl(pkl, model)
        pred, _ = measure(v, model)
        gt = truth[parts[0]]
        err = {k: round(pred[k] - gt[k], 1) for k in keys}
        photos[d] = {"subject": parts[0], "setting": parts[1], "predicted": pred,
                     "true": {k: gt[k] for k in keys}, "signed_error_cm": err}
        rows.append((parts[0], parts[1], err, {k: gt[k] for k in keys}))
    if not rows:
        sys.exit(f"No fitted HBW photos found in {args.hbw_pred_folder}")

    def summary(sel):
        out = {}
        for k in keys:
            e = np.array([r[2][k] for r in sel]); t = np.array([r[3][k] for r in sel])
            out[k] = {"mean_abs_error_mm": round(float(np.mean(np.abs(e))) * 10, 1),
                      "mean_abs_error_pct": round(float(np.mean(np.abs(e) / t)) * 100, 1),
                      "mean_signed_error_cm": round(float(np.mean(e)), 1),
                      "within_10pct": f"{int(np.sum(np.abs(e) / t <= 0.10))}/{len(e)}"}
        return out

    groups = {"all": rows, "lab": [r for r in rows if r[1] == "lab"], "wild": [r for r in rows if r[1] == "wild"]}
    summ = {g: summary(r) for g, r in groups.items() if r}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"method": METHOD, "body_model": args.gender, "summary": summ, "photos": photos}, f, indent=2)
    csv_path = os.path.splitext(args.out)[0] + ".csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["photo", "subject", "setting"] + [f"pred_{k}" for k in keys] + [f"true_{k}" for k in keys]
                   + [f"error_{k}" for k in keys])
        for name, p in photos.items():
            w.writerow([name, p["subject"], p["setting"]] + [p["predicted"][k] for k in keys]
                       + [p["true"][k] for k in keys] + [p["signed_error_cm"][k] for k in keys])

    print(f"\nHBW: {len(rows)} photos of {len(set(r[0] for r in rows))} people  ({args.gender} body model)")
    print("=" * 74)
    print(f"{'Group':<6}{'Measurement':<11}{'Mean error':>12}{'Mean error':>12}{'Too big(+)/small(-)':>21}{'Within 10%':>12}")
    print(f"{'':<17}{'(mm)':>12}{'(%)':>12}{'(cm, average)':>21}")
    print("-" * 74)
    for g, sm in summ.items():
        for k in keys:
            x = sm[k]
            print(f"{g:<6}{k[:-3]:<11}{x['mean_abs_error_mm']:>12.1f}{x['mean_abs_error_pct']:>12.1f}"
                  f"{x['mean_signed_error_cm']:>+21.1f}{x['within_10pct']:>12}")
        print("-" * 74)
    print(f"[ok] saved -> {args.out} and {csv_path}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Measure SMPL-X bodies the HBW way (one script for everything).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pkl", help="SMPLify-X result file (output\\results\\<name>\\000.pkl)")
    src.add_argument("--tpose", help="body already in T-pose (.npy or .obj, 10475 points)")
    src.add_argument("--hbw_gt_folder", help="HBW smplx\\val folder: measures all true bodies")
    src.add_argument("--hbw_pred_folder", help="SMPLify-X results folder: measures all fitted HBW photos")
    ap.add_argument("--hbw_gt", help="true-body measurements made with --hbw_gt_folder")
    ap.add_argument("--gender", choices=["male", "female", "neutral"], default="neutral",
                    help="body model used (for --pkl: the gender used in SMPLify-X)")
    ap.add_argument("--models", default=os.path.join("models", "smplx"))
    ap.add_argument("--subject_id")
    ap.add_argument("--gt", help="tape-measure CSV (your own subjects)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true", help="allow replacing --out if it exists")
    args = ap.parse_args()

    model = load_model(args.models, args.gender)

    if args.hbw_gt_folder:
        files = sorted(glob.glob(os.path.join(args.hbw_gt_folder, "*.npy")))
        if not files:
            sys.exit(f"No .npy files in {args.hbw_gt_folder}")
        allm = {}
        for f in files:
            sid = os.path.splitext(os.path.basename(f))[0]
            pred, det = measure(load_vertices(f), model)
            allm[sid] = {**pred, **det}
            print(f"HBW {sid}: " + ", ".join(f"{k[:-3]} {v}" for k, v in pred.items()))
        save(args.out, {"method": METHOD, "source": args.hbw_gt_folder, "bodies": allm}, args.overwrite)
        return

    if args.hbw_pred_folder:
        measure_hbw_predictions(args, model)
        return

    if args.pkl:
        v, betas = tpose_from_pkl(args.pkl, model)
        source = {"pkl": args.pkl, "betas": [round(float(b), 4) for b in betas]}
    else:
        v = load_vertices(args.tpose)
        source = {"tpose": args.tpose}

    pred, det = measure(v, model)
    data = {"subject_id": args.subject_id, "script": "measure_body.py", "method": METHOD,
            "body_model": args.gender, "source": source, "predicted": pred, "details": det}
    err = {}
    if args.gt and args.subject_id:
        gt = load_gt(args.gt, args.subject_id)
        err = errors(pred, gt)
        data["ground_truth"] = {k: v for k, v in gt.items() if k.endswith("_cm")}
        data["note"] = "tape hip was taken at the widest part: compare it with hip_widest_cm"
        data["errors"] = err
    print(f"\n{args.subject_id or ''}  ({args.gender} body)")
    print_table(pred, err)
    save(args.out, data, args.overwrite)


if __name__ == "__main__":
    main()
