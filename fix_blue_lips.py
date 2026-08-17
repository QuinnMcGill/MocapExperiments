import argparse
import cv2
import numpy as np
import os

# ==== Parse command-line arguments ==== #
parser = argparse.ArgumentParser(description="Visualize ARKit mocap data")
parser.add_argument("--v", help="Video file name in data folder", default="tc6.mp4")

args = parser.parse_args()

# ==== Variable Initialization ==== #
tc_name = args.v.split(".")[0]
input_video_path = "data/" + args.v
print("Input path:", input_video_path)
output_video_path = "data/editted_vids/" + tc_name + "_editted.mp4"

cap = cv2.VideoCapture(input_video_path)

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

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    green_mask = cv2.inRange(
        hsv,
        np.array([35, 50, 30]),
        np.array([90, 255, 255])
    )

    # Change hue to red while preserving saturation/value
    hsv_modified = hsv.copy()

    # OpenCV hue range is 0-179.
    hsv_modified[..., 0][green_mask > 0] = 170
    hsv_modified[..., 1][green_mask > 0] = 180

    frame_modified = cv2.cvtColor(
        hsv_modified,
        cv2.COLOR_HSV2BGR
    )

    video_writer.write(frame_modified)

cap.release()
video_writer.release()
cv2.destroyAllWindows()