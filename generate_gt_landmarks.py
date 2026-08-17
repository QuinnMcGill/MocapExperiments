import cv2
import argparse
import os
import numpy as np
import pandas as pd
import csv

# Parse command line arguments
parser = argparse.ArgumentParser(description="Generate ground truth landmark positions for test frames")
parser.add_argument("--v", help="Video file located in data folder", required=True)
parser.add_argument("--g", help="Generate new frame indices", default=1, type=int)
args = parser.parse_args()

CYAN_BGR = (255, 255, 0)

# ====== Create test frames from video ====== # 

video_path = "data/" + args.v
frame_indices = []

if args.g == 1:
    cv2.namedWindow("Capture Frames to Test Landmark Accuracy", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Capture Frames to Test Landmark Accuracy", 640, 1440)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = max(1, int(1000 / fps))  # milliseconds per frame

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        cv2.imshow("Capture Frames to Test Landmark Accuracy", frame)

        key = cv2.waitKey(delay) & 0xFF

        if key == ord(' '):  # Space bar
            # CAP_PROP_POS_FRAMES returns the index of the NEXT frame after read(),
            # so subtract 1 to get the frame currently being displayed.
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            print(f"Frame index: {frame_idx}")
            frame_indices.append(frame_idx)
        elif key == ord('q') or key == 27:  # 'q' or Esc
            break

    cap.release()
    cv2.destroyAllWindows()
else:
    # Grab the test frame indices from the user
    frame_indices_input = input("Enter the frame indices to process (comma-separated): ")
    frame_indices = [int(idx.strip()) for idx in frame_indices_input.split(',') if idx.strip().isdigit()]


print(f"Frame indices to process: {frame_indices}")

# ====== Generate ground truth landmarks for the selected frames ====== #

LANDMARKS = [
    ("outer_left_corner", "Left outer corner"),
    ("inner_left_corner", "Left inner corner"),
    ("outer_right_corner", "Right outer corner"),
    ("inner_right_corner", "Right inner corner"),
    ("cupids_bow_center", "Center of cupid's bow"),
    ("left_cupids_peak", "Left cupid's bow peak"),
    ("right_cupids_peak", "Right cupid's bow peak"),
    ("top_inner_center", "Center of upper lip, inner contour"),
    ("bottom_inner_center", "Center of bottom lip, inner contour"),
    ("bottom_outer_center", "Center of bottom lip, outer contour"),
]

current_click = None

def mouse_callback(event, x, y, flags, param):
    global current_click

    if event == cv2.EVENT_LBUTTONDOWN:
        current_click = (x, y)


cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("Could not open video.")

image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cv2.namedWindow("Annotator", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Annotator", 640, 1440)
cv2.setMouseCallback("Annotator", mouse_callback)

annotations = {}

for frame_idx in frame_indices:

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = cap.read()

    if not success:
        print(f"Could not read frame {frame_idx}")
        continue

    frame_accepted = False
    frame_annotations = []

    while not frame_accepted:

        frame_annotations = []
        redo_frame = False

        # -------------------------------------------------------
        # Collect all landmarks for this frame
        # -------------------------------------------------------

        for landmark_key, landmark_description in LANDMARKS:

            current_click = None
            point_accepted = False

            while not point_accepted:

                display = frame.copy()

                cv2.putText(
                    display,
                    f"Click: {landmark_description}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    CYAN_BGR,
                    2,
                )

                cv2.putText(
                    display,
                    "SPACE/ENTER = accept   R = redo point   Q = quit",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    CYAN_BGR,
                    2,
                )

                # Previously accepted landmarks
                for pt in frame_annotations:
                    cv2.circle(display, pt, 5, (0, 255, 0), -1)

                # Proposed landmark
                if current_click is not None:
                    cv2.circle(display, current_click, 6, (255, 0, 255), -1)

                cv2.imshow("Annotator", display)

                key = cv2.waitKey(20) & 0xFF

                if key in (13, 32) and current_click is not None:
                    frame_annotations.append(current_click)
                    point_accepted = True

                elif key == ord('r'):
                    current_click = None

                elif key in (ord('q'), 27):
                    cap.release()
                    cv2.destroyAllWindows()
                    raise SystemExit

        # -------------------------------------------------------
        # Review completed frame
        # -------------------------------------------------------

        review_complete = False

        while not review_complete:

            display = frame.copy()

            for pt in frame_annotations:
                cv2.circle(display, pt, 5, (0, 255, 0), -1)

            cv2.putText(
                display,
                "ENTER = accept frame    R = redo frame",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                CYAN_BGR,
                2,
            )

            cv2.imshow("Annotator", display)

            key = cv2.waitKey(20) & 0xFF

            if key in (13, 32):
                frame_accepted = True
                review_complete = True

            elif key == ord('r'):
                redo_frame = True
                review_complete = True

            elif key in (ord('q'), 27):
                cap.release()
                cv2.destroyAllWindows()
                raise SystemExit

        # If the user requested a redo, this loop naturally
        # starts over and recollects all 10 landmarks.
        if redo_frame:
            continue

    # Save annotations only after the frame has been accepted.
    annotations[frame_idx] = {
        key: pt
        for (key, _), pt in zip(LANDMARKS, frame_annotations)
    }

cap.release()
cv2.destroyAllWindows()

print("\nAnnotations:\n")

for frame_idx, pts in annotations.items():
    print(f"\nFrame {frame_idx}")
    for name, pt in pts.items():
        print(f"  {name:28s}: {pt}")

# ====== Save the annotations to a CSV file ====== #

output_dir = "gt_landmarks"
os.makedirs(output_dir, exist_ok=True)

vid_name = os.path.splitext(os.path.basename(video_path))[0]
csv_path = os.path.join(output_dir, f"{vid_name}_gt_landmarks.csv")

# Build CSV Header
header = [
    "frame_idx",
    "image_width",
    "image_height",
]

for key, _ in LANDMARKS:
    header.extend([f"{key}_x", f"{key}_y"])

# Write entries to CSV file
with open(csv_path, "w", newline="") as f:

    writer = csv.writer(f)
    writer.writerow(header)

    for frame_idx in frame_indices:

        pts = annotations[frame_idx]

        row = [
            frame_idx,
            image_width,
            image_height,
        ]

        for key, _ in LANDMARKS:
            x, y = pts[key]
            row.extend([x, y])

        writer.writerow(row)

print(f"\nSaved annotations to:\n{csv_path}")