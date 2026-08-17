import os
import cv2
import argparse
from matplotlib.path import Path
import pandas as pd

# ==========================================================
# Command-line arguments
# ==========================================================

parser = argparse.ArgumentParser()
parser.add_argument("--v", required=True,
                    help="Video in data/ to process")
args = parser.parse_args()

video_name = args.v.split(".mp4")[0]
original_video = os.path.join("data", args.v)
mediapipe_dir = os.path.join("MediaPipe", "video_outputs")
openface_dir = os.path.join("OpenFace2.0", "video_outputs")

mocal_vids = [os.path.join(mediapipe_dir, video_name + "_mediapipe.mp4"),
              os.path.join(openface_dir, video_name + "_openface.mp4")]

gt_csv = os.path.join(
    "gt_landmarks",
    f"{video_name}_gt_landmarks.csv"
)

output_dir = "comparison_frames"
os.makedirs(output_dir, exist_ok=True)

TEST_FRAMES = [143, 386, 577, 756, 940, 1091, 1233]

GT_COLOR = (255, 0, 0)      # BLUE in BGR
GT_RADIUS = 5
GT_THICKNESS = -1

# ==========================================================
# Load GT landmarks
# ==========================================================

df = pd.read_csv(gt_csv)

ground_truth = {}

for _, row in df.iterrows():

    frame_idx = int(row["frame_idx"])

    pts = []

    for col in df.columns:
        if col.endswith("_x"):
            y_col = col[:-2] + "_y"
            pts.append((int(row[col]), int(row[y_col])))

    ground_truth[frame_idx] = pts
    
# ==========================================================
# Open video
# ==========================================================

cv2.namedWindow("Bounding Box Selection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Bounding Box Selection", 640, 1440)
cap = cv2.VideoCapture(original_video)

if not cap.isOpened():
    raise RuntimeError(f"Could not open {original_video}")

crop_boxes = {}

cv2.namedWindow("Bounding Box Selection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Bounding Box Selection", 640, 1440)

for frame_idx in TEST_FRAMES:

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = cap.read()

    if not success:
        print(f"Couldn't read frame {frame_idx}")
        continue

    display = frame.copy()

    # Overlay GT landmarks
    for pt in ground_truth[frame_idx]:
        cv2.circle(
            display,
            pt,
            GT_RADIUS,
            GT_COLOR,
            GT_THICKNESS,
        )

    roi = cv2.selectROI(
        "Bounding Box Selection",
        display,
        showCrosshair=True,
        fromCenter=False,
    )

    crop_boxes[frame_idx] = roi

cap.release()
cv2.destroyWindow("Bounding Box Selection")

# ==========================================================
# Generate cropped comparison images
# ==========================================================

for mocap_video in mocal_vids:

    cap = cv2.VideoCapture(mocap_video)

    if not cap.isOpened():
        print(f"Couldn't open {mocap_video}")
        continue

    method_name = os.path.splitext(os.path.basename(mocap_video))[0]

    for frame_idx in TEST_FRAMES:

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        success, frame = cap.read()

        if not success:
            print(f"Couldn't read frame {frame_idx} from {method_name}")
            continue

        # Overlay GT landmarks
        for pt in ground_truth[frame_idx]:
            cv2.circle(
                frame,
                pt,
                GT_RADIUS,
                GT_COLOR,
                GT_THICKNESS,
            )

        x, y, w, h = crop_boxes[frame_idx]

        cropped = frame[y:y+h, x:x+w]

        out_path = os.path.join(
            output_dir,
            f"{method_name}_frame_{frame_idx}.png"
        )

        cv2.imwrite(out_path, cropped)

    cap.release()

print(f"\nSaved comparison images to {output_dir}")