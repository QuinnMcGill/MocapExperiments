import cv2
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import mediapipe as mp
import pandas as pd
import argparse

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import numpy as np
import matplotlib.pyplot as plt

import mp_helpers as mph

# ==== Parse command-line arguments ==== #
parser = argparse.ArgumentParser(description="Visualize ARKit mocap data")
parser.add_argument("--v", help="Path to video file", default="data/tc6.mp4")
parser.add_argument("--c", help="Path to csv file (to be created)", default="MediaPipe/results/testcase6.csv")
parser.add_argument("--bs", help="Add the blendshapes to the output video", default=0, type=int)

args = parser.parse_args()

# ==== Variable Initialization ==== #
# ---- Formating parameters ---- #
start_x = 20
start_y = 30
line_height = 30

max_rows_per_col = 31  # controls column height
col_width = 290       # horizontal spacing between columns

# ---- Image display parameters ---- #
max_width = 1280
max_height = 720

# ---- Control paramters ---- #
frame_idx = 0
earlyStop = False

# ---- Variables for output video ---- #
tc_name = args.v.split("/")[-1].split(".")[0]
output_video_path = "MediaPipe/video_outputs/" + tc_name + "_gray_mediapipe.mp4"

# ==== Tracking with MediaPipe ==== #
# Load the .task model and set up the options
base_options = python.BaseOptions(model_asset_path="MediaPipe/models/face_landmarker_v2_with_blendshapes.task")
options = vision.FaceLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO, output_face_blendshapes=True, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# Load the video file
cv2.namedWindow("MediaPipe Facial Motion Capture", cv2.WINDOW_NORMAL)
cv2.resizeWindow("MediaPipe Facial Motion Capture", 640, 1440)
cap = cv2.VideoCapture(args.v)

# Get input video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create MP4 writer
fourcc = cv2.VideoWriter.fourcc(*"mp4v")
video_writer = cv2.VideoWriter(
    output_video_path,
    fourcc,
    fps,
    (frame_width, frame_height)
)

print("Output path:", output_video_path)
print("VideoWriter opened:", video_writer.isOpened())
mp_results_list = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp_ms = int((frame_idx / fps) * 1000)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    blendshape_cols = []  # Initialize blendshape columns for this frame

    # ===== Extract blendshapes ===== #
    if result.face_blendshapes:
        blendshapes = result.face_blendshapes[0]

        row_dict = {"frame": frame_idx}

        for bs in blendshapes:
            row_dict[bs.category_name] = bs.score

        # Only define this when blendshapes exist
            blendshape_cols = [bs.category_name for bs in blendshapes]
    else:
        row_dict = {"frame": frame_idx}

    mp_results_list.append(row_dict)                # for the total dataframe
    temp_df = temp_df = pd.DataFrame([row_dict])    # convenient format for single frame

    # ==== Visualize the tracking ==== #
    # ---- Landmarks and mesh ---- #
    annotated_image = mph.draw_mediapipe_lip_contours(mp_image.numpy_view(), result, contour_color=mph.WHITE_COLOR)
    annotated_image = mph.draw_mediapipe_lip_contour_landmarks(annotated_image, result, circle_radius=4, landmark_color=mph.RED_COLOR)
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    # ---- Add blendshapes to output video ---- #
    if args.bs == 1:
        h, w = annotated_image.shape[:2]

        # Scale relative to 1080p
        scale = h / 1080

        font_scale = min(1.5, max(0.5, 0.7 * scale))
        thickness = max(1, int(2 * scale))
        line_height = int(30 * scale)
        col_width = int(260 * scale)

        start_x = int(20 * scale)
        start_y = int(40 * scale)

        row = temp_df.iloc[0]

        for i, col in enumerate(blendshape_cols):
            value = row[col]
            abs_val = abs(value)

            # Color logic (BGR format)
            if abs_val > 0.5:
                color = (0, 255, 0)       # green
            elif abs_val > 0.25:
                color = (0, 255, 255)     # yellow
            else:
                color = (0, 0, 0)   # black

            # Column positioning
            col_idx = i // max_rows_per_col            
            row_idx = i % max_rows_per_col

            if col_idx == 1:
                x = w - start_x - col_width
            else:
                x = start_x + col_idx * col_width
            y = start_y + row_idx * line_height

            text = f"{col}: {value:.2f}"

            cv2.putText(annotated_image, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, color, thickness)
        
    cv2.imshow("MediaPipe Facial Motion Capture", annotated_image)
    video_writer.write(annotated_image)

    # ==== Flow control ==== #
    key = cv2.waitKey(5) & 0xFF

    # ESC to quit
    if key == 27:
        earlyStop = True
        break

    # SPACE to pause
    if key == ord(' '):
        print("Paused. Press SPACE to resume.")

        while True:
            pause_key = cv2.waitKey(10) & 0xFF

            if pause_key == ord(' '):   # resume
                print("Resumed.")
                break
            elif pause_key == 27:       # allow exit while paused
                earlyStop = True
                break

    frame_idx += 1

# ==== Save the results ==== #
if earlyStop == False:
    mp_df = pd.DataFrame(mp_results_list)
    mp_df.fillna(0, inplace=True)
    mp_df.to_csv(args.c)

cap.release()
video_writer.release()
cv2.destroyAllWindows()