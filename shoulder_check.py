r"""
shoulder_check.py - check S03's shoulder width four ways.

READ-ONLY: it does not change any results file. It only prints numbers
and saves a picture (shoulder_check_S03_<gender>.png).

Run from the project folder:
  python shoulder_check.py --mesh output\meshes\S03_Front\000.obj --gender male
  python shoulder_check.py --mesh output_female\meshes\S03_Front\000.obj --gender female

The four ways:
  OLD      - what extract_measurements.py does now (vertices 3014 and 6159)
  JOINTS   - straight line between the two shoulder joints of the SMPL-X body
  BONES    - straight line between the tops of the two shoulder bones (acromion)
  BACK     - the same two shoulder bones, but following the curve of the upper back,
             like a tape measure laid over the back
"""
import argparse
import csv
import os
import numpy as np

OLD_LEFT, OLD_RIGHT = 3014, 6159        # the points the current script uses
LEFT_SHOULDER_JOINT, RIGHT_SHOULDER_JOINT = 16, 17   # SMPL-X joint order


def load_obj(path):
    verts = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(verts, dtype=np.float64)


def load_regressor(model_path):
    data = np.load(model_path, allow_pickle=True)
    reg = data["J_regressor"]
    if reg.dtype == object:                      # stored as a sparse matrix
        reg = reg.item()
        reg = reg.toarray() if hasattr(reg, "toarray") else np.asarray(reg)
    return np.asarray(reg, dtype=np.float64)


def tape_value(csv_path, subject):
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("subject_id", "").strip() == subject:
                return float(row["shoulder_cm"])
    return None


def acromion(v, joint):
    """Top of the shoulder bone: the highest surface point directly above the shoulder joint
    (from the joint up to 3 cm further out). Inner points are excluded so the neck slope is
    not picked; points further out are excluded so the arm is not picked."""
    side = np.sign(joint[0])
    dx = (v[:, 0] - joint[0]) * side            # 0 = above the joint, positive = further out
    near = (dx >= 0) & (dx < 0.03) & (np.abs(v[:, 2] - joint[2]) < 0.06) \
           & (v[:, 1] > joint[1]) & (v[:, 1] < joint[1] + 0.12)
    idx = np.where(near)[0]
    if len(idx) == 0:
        return None, None
    best = idx[np.argmax(v[idx, 1])]
    return best, v[best]


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def back_curve(v, a, b, front_sign):
    """Length of a TAUT tape over the back between points a and b, at shoulder height.
    A taut tape does not dip into hollows, so it follows the outer (convex) edge of the back."""
    y = (a[1] + b[1]) / 2
    band = v[np.abs(v[:, 1] - y) < 0.015]
    x_lo, x_hi = sorted([a[0], b[0]])
    pts = band[(band[:, 0] > x_lo) & (band[:, 0] < x_hi)][:, [0, 2]]
    pts = np.vstack([pts, [a[0], a[2]], [b[0], b[2]]])
    pts[:, 1] *= front_sign                     # now the back is always the low side
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    hull = []                                   # lower convex chain = taut tape over the back
    for p in pts:
        while len(hull) >= 2 and _cross(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)
    hull = np.array(hull)
    hull[:, 1] *= front_sign
    if len(hull) < 2:
        return None, None
    return float(np.sum(np.linalg.norm(np.diff(hull, axis=0), axis=1))), hull


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--gender", choices=["male", "female", "neutral"], required=True)
    ap.add_argument("--subject", default="S03")
    ap.add_argument("--models", default=os.path.join("models", "smplx"))
    ap.add_argument("--gt", default=os.path.join("ground_truth", "measurements.csv"))
    args = ap.parse_args()

    v = load_obj(args.mesh)
    if len(v) != 10475:
        raise SystemExit(f"This is not an SMPL-X mesh ({len(v)} vertices).")
    model_file = os.path.join(args.models, f"SMPLX_{args.gender.upper()}.npz")
    J = load_regressor(model_file) @ v
    jl, jr = J[LEFT_SHOULDER_JOINT], J[RIGHT_SHOULDER_JOINT]

    # which way is the front? the head leans towards the face side
    head = J[15]
    front_sign = 1.0 if head[2] >= J[12][2] - 0.02 else -1.0

    il, al = acromion(v, jl)
    ir, ar = acromion(v, jr)

    old = np.linalg.norm(v[OLD_LEFT] - v[OLD_RIGHT]) * 100
    joints = np.linalg.norm(jl - jr) * 100
    bones = np.linalg.norm(al - ar) * 100 if al is not None and ar is not None else None
    back, path = back_curve(v, al, ar, front_sign) if bones is not None else (None, None)
    back = back * 100 if back is not None else None
    tape = tape_value(args.gt, args.subject)

    print(f"\n{args.subject} shoulder check - {args.gender} body - {args.mesh}")
    print("=" * 66)
    print(f"{'Method':<44}{'Value':>9}{'Error':>12}")
    print("-" * 66)

    def row(name, val):
        if val is None:
            print(f"{name:<44}{'n/a':>9}")
            return
        err = f"{(val - tape) / tape * 100:+.1f}%" if tape else ""
        print(f"{name:<44}{val:>7.1f} cm{err:>10}")

    row("OLD   (current script: chin to chest)", old)
    row("JOINTS (straight, joint to joint)", joints)
    row("BONES (straight, shoulder bone to bone)", bones)
    row("BACK  (along the back, bone to bone)", back)
    if tape:
        print("-" * 66)
        print(f"{'TAPE measure':<44}{tape:>7.1f} cm")
    print("=" * 66)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 6))
        a1.scatter(v[:, 0], v[:, 1], s=0.3, c="#9aa4b1")
        a1.plot(*v[[OLD_LEFT, OLD_RIGHT]][:, :2].T, "o-", c="#e03131", label="OLD")
        a1.plot([jl[0], jr[0]], [jl[1], jr[1]], "s-", c="#5c7cfa", label="JOINTS")
        if bones is not None:
            a1.plot([al[0], ar[0]], [al[1], ar[1]], "^-", c="#2b8a3e", label="BONES")
        a1.set_title("Front view"); a1.set_aspect("equal"); a1.legend(loc="lower right")
        top = v[np.abs(v[:, 1] - (al[1] + ar[1]) / 2) < 0.03] if bones is not None else v
        a2.scatter(top[:, 0], top[:, 2], s=2, c="#9aa4b1")
        if path is not None:
            a2.plot(path[:, 0], path[:, 1], c="#d9822b", lw=2.5, label="BACK (tape over the back)")
            a2.plot([al[0], ar[0]], [al[2], ar[2]], "^-", c="#2b8a3e", label="BONES (straight)")
        a2.set_title("Seen from above, at shoulder height"); a2.set_aspect("equal"); a2.legend(loc="lower right")
        for ax in (a1, a2):
            ax.set_xticks([]); ax.set_yticks([])
        out = f"shoulder_check_{args.subject}_{args.gender}.png"
        plt.tight_layout(); plt.savefig(out, dpi=110)
        print(f"Picture saved: {out}")
    except Exception as e:
        print(f"(picture skipped: {e})")


if __name__ == "__main__":
    main()
