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

# ---- Parse command-line arguments ---- #
parser = argparse.ArgumentParser(description="Visualize ARKit mocap data")
parser.add_argument("--v", help="Path to video file", default="ARKit/data/testcase4/Testcase4_2_iPhone.mov")
parser.add_argument("--c", help="Path to the calibrated .csv file of mocap", default="ARKit/data/testcase4/Testcase4_2_iPhone_cal.csv")

args = parser.parse_args()

# ---- Run landmark detection ---- #

# Lost the .task model and set up the options
base_options = python.BaseOptions(model_asset_path="MediaPipe/models/face_landmarker_v2_with_blendshapes.task")
options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# Load  the video file
cap = cv2.VideoCapture(args.v)
print(cap)
mp_results_list = []
frame_idx = 0

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

    result = detector.detect(mp_image, frame_idx)

    # ===== Extract blendshapes =====
    if result.face_blendshapes:
        blendshapes = result.face_blendshapes[0]

        row_dict = {"frame": frame_idx}

        for bs in blendshapes:
            row_dict[bs.category_name] = bs.score
    else:
        row_dict = {"frame": frame_idx}

    mp_results_list.append(row_dict)

    frame_idx += 1

mp_df = pd.DataFrame(mp_results_list)
mp_df.fillna(0, inplace=True)

print(mp_df.head(10))