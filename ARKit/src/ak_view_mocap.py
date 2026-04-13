import os
import argparse

import pandas as pd

import matplotlib.pyplot as plt
import cv2

# ---- Parse command-line arguments ---- #
parser = argparse.ArgumentParser(description="Visualize ARKit mocap data")
parser.add_argument("--v", help="Path to video file", default="ARKit/data/testcase4/Testcase4_2_iPhone.mov")
parser.add_argument("--c", help="Path to the calibrated .csv file of mocap", default="ARKit/data/testcase4/Testcase4_2_iPhone_cal.csv")

args = parser.parse_args()

# ---- Import blendshape weights from .csv file ---- #

df = pd.read_csv(args.c)
blendshape_cols = df.columns[2:]  
print("Number of Blendshape columns:", len(blendshape_cols))

# ---- Play video ---- #

cap = cv2.VideoCapture(args.v)
cv2.namedWindow("Video with ARKit Data", cv2.WINDOW_NORMAL)

# Control variables
frame_idx = 0
earlyStop = False

# Formating and display variables
start_x = 20
start_y = 30
line_height = 30

max_rows_per_col = 31  # controls column height
col_width = 290       # horizontal spacing between columns

while cap.isOpened() and earlyStop == False:
    ret, frame = cap.read()
    if not ret:
        break

    row = df.iloc[frame_idx]

    rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    height, width = rotated.shape[:2]

    # ===== Display all blendshapes =====

    for i, col in enumerate(blendshape_cols):
        value = row[col]
        abs_val = abs(value)

        # Color logic (BGR format)
        if abs_val > 0.5:
            color = (0, 255, 0)       # green
        elif abs_val > 0.25:
            color = (0, 255, 255)     # yellow
        else:
            color = (0, 0, 0)   # white

        # Column positioning
        col_idx = i // max_rows_per_col            
        row_idx = i % max_rows_per_col

        if col_idx == 1:
            x = width - start_x - col_width
        else:
            x = start_x + col_idx * col_width
        y = start_y + row_idx * line_height

        text = f"{col}: {value:.2f}"

        cv2.putText(rotated, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)

    resized = cv2.resize(rotated, (int(width), int(height)))

    cv2.imshow("Video with ARKit Data", resized)

    key = cv2.waitKey(5) & 0xFF

    # ESC to quit
    if key == 27:
        earlyStop = True

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

cap.release()
cv2.destroyAllWindows()


