import cv2
import numpy as np
import cv2
import os
import numpy as np
from pathlib import Path
import tqdm
import time

# Paths for dataset
input_path = "data/MayaFaces"
output_path = "data/MayaFaces_editted"

image_path = os.path.join(input_path, "images")
ul_path = os.path.join(input_path, "ul_masks")
ll_path = os.path.join(input_path, "ll_masks")

# Edit Maya Face Images
maya_faces = sorted(os.listdir(image_path))
for maya_face in maya_faces:
    if not maya_face.lower().endswith(".png"):
            continue
    
    # Access mask image
    img_path = os.path.join(image_path, maya_face)
    img_name = maya_face.split('.')[0]
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    
    # Split channels
    rgb = img[:, :, :3].astype(np.float32)
    alpha = img[:, :, 3:4].astype(np.float32) / 255.0

    # Mid-gray background
    gray_value = 128

    background = np.full(
        rgb.shape,
        gray_value,
        dtype=np.float32
    )

    # Alpha compositing
    rgb_no_alpha = (
        rgb * alpha
        + background * (1.0 - alpha)
    )

    rgb_no_alpha = np.clip(
        rgb_no_alpha,
        0,
        255
    ).astype(np.uint8)

    output_file = os.path.join(output_path, image_path.split('/')[-1], maya_face)

    # Save result
    cv2.imwrite(output_file, rgb_no_alpha)    


# Create Binary Masks for Upper and Loewr Lips
for mask_path in [ul_path, ll_path]:
    mask_files = sorted(os.listdir(mask_path), key=str.lower)
    for filename in mask_files:
        if not filename.lower().endswith(".png"):
            continue

        # Access mask image
        img_path = os.path.join(mask_path, filename)
        img_name = filename.split('.')[0]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        output_file = output_path + mask_path.split('/')[-1] + '/' + filename
        output_file = os.path.join(output_path, mask_path.split('/')[-1], filename)

        # Editing and saving mask #
        # Use one RGB channel (all are identical)
        gray = img[:, :, 0]
        mask = np.where(gray == 207, 255, 0).astype(np.uint8)

        # Save Result
        cv2.imwrite(output_file, mask)
