import cv2
import numpy as np
import cv2
import numpy as np
from pathlib import Path

# Path to image
image_path = "/home/quinnm/repo/MocapExperiments/data/MayaFaces/images/img_0000.png"

# Read image unchanged (preserves alpha channel if present)
img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

if img is None:
    raise ValueError(f"Could not read image: {image_path}")

# Basic info
print("==== Maya Image ====")
print(f"Shape: {img.shape}")
print(f"Data type: {img.dtype}")
print(f"Size (HxW): {img.shape[:2]}")

# Number of channels
if len(img.shape) == 2:
    channels = 1
elif len(img.shape) == 3:
    channels = img.shape[2]
else:
    channels = -1

print(f"Channels: {channels}")

# Pixel statistics
print("\nPixel Statistics:")
print(f"Min value: {img.min()}")
print(f"Max value: {img.max()}")
print(f"Mean value: {img.mean():.2f}")
print(f"Unique values: {np.unique(img)}")

pixels = img.reshape(-1, 4)
unique_pixels = np.unique(pixels, axis=0)

# Per-channel statistics
if channels > 1:
    print("\nPer-channel statistics:")
    for c in range(channels):
        print(
            f"Channel {c}: "
            f"min={img[:,:,c].min()}, "
            f"max={img[:,:,c].max()}, "
            f"mean={img[:,:,c].mean():.2f}"
        )

raise SystemExit
# --- Editting the image for inary segmentation --- #
# Use one RGB channel (all are identical)
gray = img[:, :, 0]

# Only keep the lip region
output_file = "temp_binary.png"
mask = np.where(gray == 207, 255, 0).astype(np.uint8)

cv2.imwrite(output_file, mask)

img_out = cv2.imread(output_file, cv2.IMREAD_UNCHANGED)

print("\n==== Binary Mask ====")
print(f"Shape: {img_out.shape}")
print(f"Data type: {img_out.dtype}")
print(f"Size (HxW): {img_out.shape[:2]}")

# Number of channels
if len(img_out.shape) == 2:
    channels = 1
elif len(img_out.shape) == 3:
    channels = img_out.shape[2]
else:
    channels = -1

print(f"Channels: {channels}")

# Pixel statistics
print("\nPixel Statistics:")
print(f"Min value: {img_out.min()}")
print(f"Max value: {img_out.max()}")
print(f"Mean value: {img_out.mean():.2f}")
print(f"Unique values: {np.unique(img_out)}")