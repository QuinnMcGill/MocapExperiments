import argparse
import cv2
import numpy as np
import os
from utils import *

# ==== Parse command-line arguments ==== #
parser = argparse.ArgumentParser(description="Generate the \"gold standard\" lip masks using filtering")
parser.add_argument("--v", help="Video file name in data folder", default="tc6.mp4")
parser.add_argument("--display_masks", help="Display ground truth masks", type=int, default=1)

args = parser.parse_args()

# ==== Helper Functions ==== #
def display_masks(original_frame, upper_mask, lower_mask, window_name="Lip Masks"):
    """
    Display the original frame alongside a color visualization
    of the upper and lower lip masks.

    Upper lip  -> red
    Lower lip  -> green
    """

    # Create black RGB/BGR visualization
    mask_visualization = np.zeros_like(original_frame)

    # Upper lip -> red
    mask_visualization[upper_mask > 0] = (0, 0, 255)

    # Lower lip -> green
    mask_visualization[lower_mask > 0] = (0, 255, 0)

    # Put original and mask visualization side-by-side
    combined = np.hstack([
        original_frame,
        mask_visualization
    ])

    cv2.imshow(window_name, combined)

def save_gold_standard_masks(
    upper_masks,
    lower_masks,
    save_path
):
    """
    Save binary lip masks using flattened pixel indices and
    per-frame offsets.

    Parameters
    ----------
    upper_masks : list[np.ndarray]
        List of binary upper-lip masks, one per frame.
    lower_masks : list[np.ndarray]
        List of binary lower-lip masks, one per frame.
    save_path : str
        Path to output .npz file.
    """

    upper_indices = []
    upper_offsets = [0]

    lower_indices = []
    lower_offsets = [0]

    for upper_mask, lower_mask in zip(
        upper_masks,
        lower_masks
    ):

        # Get flattened indices of foreground pixels
        upper_idx = np.flatnonzero(
            upper_mask > 0
        ).astype(np.uint32)

        lower_idx = np.flatnonzero(
            lower_mask > 0
        ).astype(np.uint32)

        upper_indices.append(upper_idx)
        upper_offsets.append(
            upper_offsets[-1] + len(upper_idx)
        )

        lower_indices.append(lower_idx)
        lower_offsets.append(
            lower_offsets[-1] + len(lower_idx)
        )

    # Handle case where there are no masks
    if len(upper_indices) > 0:
        upper_indices = np.concatenate(
            upper_indices
        )
    else:
        upper_indices = np.array(
            [],
            dtype=np.uint32
        )

    if len(lower_indices) > 0:
        lower_indices = np.concatenate(
            lower_indices
        )
    else:
        lower_indices = np.array(
            [],
            dtype=np.uint32
        )

    upper_offsets = np.asarray(
        upper_offsets,
        dtype=np.uint64
    )

    lower_offsets = np.asarray(
        lower_offsets,
        dtype=np.uint64
    )

    np.savez_compressed(
        save_path,
        upper_indices=upper_indices,
        lower_indices=lower_indices,
        upper_offsets=upper_offsets,
        lower_offsets=lower_offsets
    )

    print(f"Saved gold standard masks to: {save_path}")

def largest_connected_component(mask):
    """
    Returns a binary mask containing only the largest connected component.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (0 and 255).

    Returns
    -------
    largest_mask : np.ndarray
        Binary mask with only the largest connected component.
    """

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    # No foreground pixels.
    if num_labels <= 1:
        return np.zeros_like(mask)

    # Ignore background (label 0).
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    largest_mask = np.zeros_like(mask)
    largest_mask[labels == largest_label] = 255

    return largest_mask

# ==== Variable Initialization ==== #
tc_name = args.v.split(".")[0]
input_video_path = "data/" + args.v
print("Input path:", input_video_path)
output_csv_path = "gs_masks/" + tc_name + "_gs_masks.csv"
output_video_path = "gs_masks/" + tc_name + "_gs_masks.mp4"
print("Output csv path:", output_csv_path)
print("Output video path: ", output_video_path)

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

if not video_writer.isOpened():
    raise RuntimeError(
        f"Could not open VideoWriter: {output_video_path}"
    )

# ------ Green Bottom Lip ------ #
# green_hsv, red1_hsv, red2_hsv = get_hsv_vals(tc_name)

# [G_h1, G_h2], [G_s1, G_s2], [G_v1, G_v2] = green_hsv

# print("Green Mask (according to online picker):") 
# print("H: ", G_h1, " -> ", G_h2)
# print("S: ", G_s1, " -> ", G_s2)
# print("V: ", G_v1, " -> ", G_v2)

# # Define Mask in OpenCV colour space (H: 0-180, S: 0-255, V: 0-255)
# green_start = np.array([round(G_h1/355*180), round(G_s1/100*255), round(G_v1/100*255)])
# green_end = np.array([round(G_h2/355*180), round(G_s2/100*255), round(G_v2/100*255)])

# # ------ Red Top Lip ------ #

# [R1_h1, R1_h2], [R1_s1, R1_s2], [R1_v1, R1_v2] = red1_hsv
# [R2_h1, R2_h2], [R2_s1, R2_s2], [R2_v1, R2_v2] = red2_hsv

# print("Red 1 Mask (according to online picker):")
# print("H: ", R1_h1, " -> ", R1_h2)
# print("S: ", R1_s1, " -> ", R1_s2)
# print("V: ", R1_v1, " -> ", R1_v2)

# print("Red 2 Mask (according to online picker):")
# print("H: ", R2_h1, " -> ", R2_h2)
# print("S: ", R2_s1, " -> ", R2_s2)
# print("V: ", R2_v1, " -> ", R2_v2)

# # Define masks in OpenCV colour space (H: 0-180, S: 0-255, V: 0-255)
# red_1_start = np.array([round(R1_h1/355*180), round(R1_s1/100*255), round(R1_v1/100*255)])
# red_1_end = np.array([round(R1_h2/355*180), round(R1_s2/100*255), round(R1_v2/100*255)])

# red_2_start = np.array([round(R2_h1/355*180), round(R2_s1/100*255), round(R2_v1/100*255)])
# red_2_end = np.array([round(R2_h2/355*180), round(R2_s2/100*255), round(R2_v2/100*255)])

# ----- Video Processing ----- #

# Lists to store the top and bottom lip masks (per-frame)
upper_masks = []
lower_masks = []

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    green_mask, red_mask = get_masks(tc_name, hsv)

    # --------------------------------------------------------
    # Remove small connected components
    # --------------------------------------------------------

    green_mask = largest_connected_component(green_mask)
    red_mask = largest_connected_component(red_mask)

    # Close any holes in the masks
    green_mask = close_mask_holes(green_mask)
    red_mask = close_mask_holes(red_mask)

    # Save masks for this frame
    upper_masks.append(red_mask)
    lower_masks.append(green_mask)

    # --------------------------------------------------------
    # Visualize masks
    # --------------------------------------------------------
    if args.display_masks == 1:
        window_name = "Gold standard lip masks"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 1440)

        display_masks(
            frame,
            red_mask,
            green_mask, 
            window_name
        )

    # --------------------------------------------------------
    # Modifying lip colour
    # --------------------------------------------------------

    # Change hue to red while preserving saturation/value
    hsv_modified = hsv.copy()

    # OpenCV hue range is 0-179.
    hsv_modified[..., 0][green_mask > 0] = 170  # maroon bottom lip
    hsv_modified[..., 1][green_mask > 0] = 180

    hsv_modified[..., 0][red_mask > 0] = 99 # blue top lip
    hsv_modified[..., 1][red_mask > 0] = 180

    # hsv_modified[..., 0][red_mask > 0] = round(180/360.0 * 180)
    # hsv_modified[..., 1][red_mask > 0] = 255
    # hsv_modified[..., 2][red_mask > 0] = 255

    frame_modified = cv2.cvtColor(
        hsv_modified,
        cv2.COLOR_HSV2BGR
    )

    video_writer.write(frame_modified)

    # --------------------------------------------------------
    # Wait for key
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()

save_path = (
    f"gs_masks/"
    f"gs_masks_{tc_name}.npz"
)

os.makedirs(
    os.path.dirname(save_path),
    exist_ok=True
)

save_gold_standard_masks(
    upper_masks,
    lower_masks,
    save_path
)