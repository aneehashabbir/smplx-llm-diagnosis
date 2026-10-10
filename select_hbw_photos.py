r"""
select_hbw_photos.py - pick the best HBW validation photos and prepare them for SMPLify-X.

For every one of the 10 validation people it picks up to 4 lab photos and 4 "in the wild"
photos where:
  - exactly the whole body is visible (head, shoulders, hips, knees, ankles all detected)
  - the person faces the camera (not turned sideways or showing the back)
and copies them, with their body points, into ONE folder that SMPLify-X can run on:

  hbw_selected\images\012_lab_00020.png
  hbw_selected\keypoints\012_lab_00020_keypoints.json

It only READS the HBW folder and never changes it.

Run from the project folder:
  python select_hbw_photos.py --hbw D:\HBW\HBW --out hbw_selected
"""
import argparse
import csv
import glob
import json
import os
import shutil
import sys

from PIL import Image

# OpenPose BODY_25 points that must be visible for a full-body photo
NOSE, NECK, R_SH, L_SH, MID_HIP = 0, 1, 2, 5, 8
R_HIP, R_KNEE, R_ANK, L_HIP, L_KNEE, L_ANK = 9, 10, 11, 12, 13, 14
NEEDED = [NOSE, NECK, R_SH, L_SH, MID_HIP, R_HIP, R_KNEE, R_ANK, L_HIP, L_KNEE, L_ANK]
MIN_CONF = 0.4


def read_person(json_path):
    with open(json_path, encoding="utf-8") as f:
        people = json.load(f).get("people", [])
    if len(people) != 1:
        return None, f"{len(people)} people"
    k = people[0]["pose_keypoints_2d"]
    pts = [(k[3 * i], k[3 * i + 1], k[3 * i + 2]) for i in range(len(k) // 3)]
    return pts, None


def score_photo(pts):
    """Return (score, reason). Higher = clearer, more frontal full-body photo."""
    if any(pts[i][2] < MIN_CONF for i in NEEDED):
        return None, "body not fully visible"
    # facing the camera: the person's right shoulder appears on the LEFT of the image
    if pts[R_SH][0] >= pts[L_SH][0]:
        return None, "seen from the back"
    shoulder_w = abs(pts[L_SH][0] - pts[R_SH][0])
    torso = abs(pts[MID_HIP][1] - pts[NECK][1])
    if torso <= 0:
        return None, "odd pose"
    frontal = shoulder_w / torso            # small = turned sideways
    if frontal < 0.45:
        return None, "turned sideways"
    conf = sum(pts[i][2] for i in NEEDED) / len(NEEDED)
    return conf * min(frontal, 1.0), "ok"


def find_image(img_dir, stem):
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        p = os.path.join(img_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hbw", required=True, help=r"the HBW folder, e.g. D:\HBW\HBW")
    ap.add_argument("--out", default="hbw_selected")
    ap.add_argument("--per_setting", type=int, default=4, help="photos per person per setting (lab / wild)")
    args = ap.parse_args()

    img_root = os.path.join(args.hbw, "images", "val")
    kp_root = os.path.join(args.hbw, "keypoints", "val")
    if not os.path.isdir(img_root) or not os.path.isdir(kp_root):
        sys.exit(f"Could not find {img_root} and {kp_root}")
    if os.path.exists(args.out):
        sys.exit(f"[stop] {args.out} already exists. Delete it or choose another --out name.")
    os.makedirs(os.path.join(args.out, "images"))
    os.makedirs(os.path.join(args.out, "keypoints"))

    rows, skipped = [], {}
    for subj_dir in sorted(os.listdir(kp_root)):
        subject = subj_dir.split("_")[0]
        for setting, folder in (("lab", "Photos_Lab"), ("wild", "Pictures_in_the_Wild")):
            kdir = os.path.join(kp_root, subj_dir, folder)
            idir = os.path.join(img_root, subj_dir, folder)
            if not os.path.isdir(kdir):
                continue
            candidates = []
            for jp in sorted(glob.glob(os.path.join(kdir, "*.json"))):
                stem = os.path.splitext(os.path.basename(jp))[0].replace("_keypoints", "")
                pts, why = read_person(jp)
                if pts is None:
                    skipped[why] = skipped.get(why, 0) + 1
                    continue
                s, why = score_photo(pts)
                if s is None:
                    skipped[why] = skipped.get(why, 0) + 1
                    continue
                img = find_image(idir, stem)
                if img is None:
                    skipped["image file missing"] = skipped.get("image file missing", 0) + 1
                    continue
                candidates.append((s, stem, jp, img, pts))
            candidates.sort(reverse=True)
            for s, stem, jp, img, pts in candidates[: args.per_setting]:
                name = f"{subject}_{setting}_{stem}"
                w, h = Image.open(img).size
                max_x = max(p[0] for p in pts if p[2] > 0)
                max_y = max(p[1] for p in pts if p[2] > 0)
                if max_x > w * 1.05 or max_y > h * 1.05:
                    skipped["points do not match image size"] = skipped.get("points do not match image size", 0) + 1
                    continue
                shutil.copy2(img, os.path.join(args.out, "images", name + os.path.splitext(img)[1]))
                shutil.copy2(jp, os.path.join(args.out, "keypoints", name + "_keypoints.json"))
                rows.append({"name": name, "subject": subject, "setting": setting,
                             "score": round(s, 3), "image_w": w, "image_h": h, "source": img})
            print(f"{subject} {setting:4s}: {min(len(candidates), args.per_setting)} picked "
                  f"from {len(candidates)} good photos")

    with open(os.path.join(args.out, "selected_photos.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["name"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nPicked {len(rows)} photos -> {args.out}\\images and {args.out}\\keypoints")
    if skipped:
        print("Photos left out: " + ", ".join(f"{k}: {v}" for k, v in skipped.items()))
    print(f"List of picked photos: {args.out}\\selected_photos.csv")


if __name__ == "__main__":
    main()
