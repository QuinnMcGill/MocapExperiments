import argparse
import cv2
import numpy as np
import os
from utils import *

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
frame_idx = -1

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    green_mask, red_mask = get_masks(tc_name, hsv)

    green_mask = largest_connected_component(green_mask)
    red_mask = largest_connected_component(red_mask)

    # Close any holes in the masks
    green_mask = close_mask_holes(green_mask)
    red_mask = close_mask_holes(red_mask)

    # Change hue to red while preserving saturation/value
    hsv_modified = hsv.copy()

    # Change the lips to a more natural colour
    hsv_modified[..., 0][green_mask > 0] = 178  
    hsv_modified[..., 1][green_mask > 0] = 165

    hsv_modified[..., 0][red_mask > 0] = 178  
    hsv_modified[..., 1][red_mask > 0] = 165

    # Brighten without overflow
    value = hsv_modified[..., 2].astype(np.int16)
    value[red_mask > 0] += 5
    value[green_mask > 0] += 50
    hsv_modified[..., 2] = np.clip(value, 0, 255).astype(np.uint8)

    frame_modified = cv2.cvtColor(
        hsv_modified,
        cv2.COLOR_HSV2BGR
    )

    video_writer.write(frame_modified)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
