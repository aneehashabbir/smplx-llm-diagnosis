import cv2
import mediapipe as mp
import numpy as np
import json
import os
import argparse


def mediapipe_to_openpose(image_path, output_dir, subject_id):
    """
    Extract 2D body keypoints using MediaPipe and convert to
    OpenPose JSON format that SMPLify-X expects.
    """
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    h, w = image.shape[:2]
    print(f"[ok] Image loaded: {w}x{h} pixels")

    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
    ) as pose:

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            print("[!] No person detected in image")
            return False

        # MediaPipe landmark indices
        # We map to OpenPose 25-keypoint format that SMPLify-X uses
        # OpenPose order: Nose, Neck, RShoulder, RElbow, RWrist,
        #                 LShoulder, LElbow, LWrist, MidHip,
        #                 RHip, RKnee, RAnkle, LHip, LKnee, LAnkle,
        #                 REye, LEye, REar, LEar,
        #                 LBigToe, LSmallToe, LHeel,
        #                 RBigToe, RSmallToe, RHeel

        MP = mp_pose.PoseLandmark

        # Mapping: OpenPose index -> MediaPipe landmark
        mapping = {
            0: MP.NOSE,
            2: MP.RIGHT_SHOULDER,
            3: MP.RIGHT_ELBOW,
            4: MP.RIGHT_WRIST,
            5: MP.LEFT_SHOULDER,
            6: MP.LEFT_ELBOW,
            7: MP.LEFT_WRIST,
            9: MP.RIGHT_HIP,
            10: MP.RIGHT_KNEE,
            11: MP.RIGHT_ANKLE,
            12: MP.LEFT_HIP,
            13: MP.LEFT_KNEE,
            14: MP.LEFT_ANKLE,
            15: MP.RIGHT_EYE,
            16: MP.LEFT_EYE,
            17: MP.RIGHT_EAR,
            18: MP.LEFT_EAR,
            19: MP.LEFT_FOOT_INDEX,
            22: MP.RIGHT_FOOT_INDEX,
        }

        lm = results.pose_landmarks.landmark

        # Compute derived keypoints
        # Neck = midpoint of shoulders
        r_shoulder = lm[MP.RIGHT_SHOULDER]
        l_shoulder = lm[MP.LEFT_SHOULDER]
        neck_x = (r_shoulder.x + l_shoulder.x) / 2
        neck_y = (r_shoulder.y + l_shoulder.y) / 2

        # MidHip = midpoint of hips
        r_hip = lm[MP.RIGHT_HIP]
        l_hip = lm[MP.LEFT_HIP]
        midhip_x = (r_hip.x + l_hip.x) / 2
        midhip_y = (r_hip.y + l_hip.y) / 2

        # Build 25-keypoint array [x, y, confidence] * 25
        keypoints = [[0.0, 0.0, 0.0]] * 25

        # Fill mapped keypoints
        for op_idx, mp_idx in mapping.items():
            lmk = lm[mp_idx]
            conf = lmk.visibility if lmk.visibility > 0 else 0.0
            keypoints[op_idx] = [
                lmk.x * w,  # pixel x
                lmk.y * h,  # pixel y
                float(conf)
            ]

        # Fill derived keypoints
        keypoints[1] = [neck_x * w, neck_y * h, 0.8]  # Neck
        keypoints[8] = [midhip_x * w, midhip_y * h, 0.8]  # MidHip

        # Flatten to 1D list [x0,y0,c0, x1,y1,c1, ...]
        flat = [val for kp in keypoints for val in kp]

        # OpenPose JSON format
        openpose_output = {
            "version": 1.3,
            "people": [{
                "person_id": [-1],
                "pose_keypoints_2d": flat,
                "face_keypoints_2d": [],
                "hand_left_keypoints_2d": [],
                "hand_right_keypoints_2d": [],
                "pose_keypoints_3d": [],
                "face_keypoints_3d": [],
                "hand_left_keypoints_3d": [],
                "hand_right_keypoints_3d": []
            }]
        }

        # Save JSON
        os.makedirs(output_dir, exist_ok=True)
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(output_dir, f"{img_name}_keypoints.json")

        with open(out_path, 'w') as f:
            json.dump(openpose_output, f, indent=2)

        print(f"[ok] Keypoints saved -> {out_path}")

        # Save visualization
        vis = image.copy()
        mp_drawing.draw_landmarks(
            vis, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        vis_path = os.path.join(output_dir, f"{img_name}_viz.jpg")
        cv2.imwrite(vis_path, vis)
        print(f"[ok] Visualization saved -> {vis_path}")

        return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Input image path")
    ap.add_argument("--output_dir", default="keypoints",
                    help="Output directory for keypoints JSON")
    ap.add_argument("--subject_id", default="S00",
                    help="Subject ID for naming")
    args = ap.parse_args()

    success = mediapipe_to_openpose(
        args.image, args.output_dir, args.subject_id)

    if success:
        print("[ok] Keypoint extraction complete")
    else:
        print("[!] Failed — check image path and try again")