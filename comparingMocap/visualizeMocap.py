import os
os.environ["QT_LOGGING_RULES"] = "*.warning=false"
import cv2
import pandas as pd
import argparse

# ==== Parse command-line arguments ==== #
parser = argparse.ArgumentParser(description="Visualize ARKit mocap data")
parser.add_argument("--tc", help="Testcase number (e.g., 1, 2, 3, 4)", default="1")
args = parser.parse_args()
tc_num = args.tc

# ==== Import the csv files of blenshape weights ==== #
arkit_csv_path = f"ARKit/data/testcase{tc_num}/Testcase{tc_num}_cal.csv"
mp_csv_path = f"MyMediaPipe/results/testcase{tc_num}.csv"

arkit_df = pd.read_csv(arkit_csv_path)
mp_df = pd.read_csv(mp_csv_path)

# ==== Grab the video file ==== #
video_path = f"ARKit/data/testcase{tc_num}/Testcase{tc_num}.mp4"
cap = cv2.VideoCapture(video_path)

paused = False
frame_idx = 0

cv2.namedWindow("ARKit", cv2.WINDOW_NORMAL)
cv2.namedWindow("MediaPipe", cv2.WINDOW_NORMAL)

# ==== Variable Initialization ==== #
# ---- Formating parameters ---- #
start_x = 20
start_y = 30
line_height = 30

max_rows_per_col = 31  # controls column height
col_width = 290       # horizontal spacing between columns

# ==== Helper Functions ==== #
def getColor(val):
    abs_val = abs(val) if value is not None else 0
    if abs_val > 0.5:
        color = (0, 255, 0)       # green
    elif abs_val > 0.25:
        color = (0, 255, 255)     # yellow
    else:
        color = (0, 0, 0)   # white
    return color

def getTextPos(i, width):
    col_idx = i // max_rows_per_col            
    row_idx = i % max_rows_per_col

    if col_idx == 1:
        x = width - start_x - col_width
    else:
        x = start_x + col_idx * col_width
    y = start_y + row_idx * line_height
    return x, y

# ==== Main Loop ==== #
while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break 

        frame_arkit = frame.copy()
        frame_mp = frame.copy()

        height, width = frame.shape[:2]

        # ---- Get rows ---- #
        if frame_idx < len(arkit_df):
            arkit_row = arkit_df.iloc[frame_idx]
        else:
            arkit_row = None

        if frame_idx < len(mp_df):
            mp_row = mp_df.iloc[frame_idx]
        else:
            mp_row = None

        # ---- Overlay text ---- #
        arkit_bs_cols = arkit_df.columns[2:]
        mp_bs_cols = mp_df.columns[3:]

        # ARKit values
        for i, col in enumerate(arkit_bs_cols):
            value = arkit_row[col] if arkit_row is not None else None
    
            # Color logic
            color = getColor(value)

            # Column positioning
            x, y = getTextPos(i, width)

            text = f"{col}: {value:.2f}"

            cv2.putText(frame_arkit, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)
            
        # MediaPipe values
        for i, col in enumerate(mp_bs_cols):
            value = mp_row[col] if mp_row is not None else None

            # Color logic
            color = getColor(value)

            # Column positioning
            x, y = getTextPos(i, width)

            text = f"{col}: {value:.2f}"

            cv2.putText(frame_mp, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)
        
        frame_idx += 1

    # ---- Show both ---- #
    cv2.imshow("ARKit", frame_arkit)
    cv2.imshow("MediaPipe", frame_mp)

    key = cv2.waitKey(10) & 0xFF

    if key == ord(' '):   # pause toggle
        paused = not paused

    elif key == 27: # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

