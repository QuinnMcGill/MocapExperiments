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
parser.add_argument("--v", help="Path to video file", default="ARKit/data/testcase4/Testcase4.mp4")
parser.add_argument("--c", help="Path to csv file (to be created)", default="MyMediaPipe/results/testcase4.csv")

args = parser.parse_args()

# ==== Variable Initialization ==== #
# ---- Formating parameters ---- #
start_x = 20
start_y = 30
line_height = 30

max_rows_per_col = 31  # controls column height
col_width = 290       # horizontal spacing between columns

# ---- Control paramters ---- #
frame_idx = 0
earlyStop = False

# ==== Tracking with MediaPipe ==== #
# Load the .task model and set up the options
base_options = python.BaseOptions(model_asset_path="MyMediaPipe/models/face_landmarker_v2_with_blendshapes.task")
options = vision.FaceLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO, output_face_blendshapes=True, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# Load the video file
# cv2.namedWindow("MediaPipe Facial Motion Capture", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("MediaPipe Facial Motion Capture", 640, 1440)
cap = cv2.VideoCapture(args.v)
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
    annotated_image = mph.draw_landmarks_on_image(mp_image.numpy_view(), result)
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    # ---- Add blendshapes ---- #
    height, width = annotated_image.shape[:2]
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
            x = width - start_x - col_width
        else:
            x = start_x + col_idx * col_width
        y = start_y + row_idx * line_height

        text = f"{col}: {value:.2f}"

        cv2.putText(annotated_image, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)
    # cv2.imshow("MediaPipe Facial Motion Capture", annotated_image)

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