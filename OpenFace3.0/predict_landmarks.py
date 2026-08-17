import cv2
import numpy as np
import os

# Import the FaceDetector and LandmarkDetector classes from the OpenFace library
from openface.face_detection import FaceDetector            #type: ignore
from openface.landmark_detection import LandmarkDetector    #type: ignore

# Import the custom utility functions for better modularity
import openface_utils as of_utils

# From CLI: openface detect "/home/quinnm/repo/MocapExperiments/data/6.jpg" --output-dir "OpenFace3.0/csv_files" --device cuda

# ===== Configuring OpenFace Models ===== #
# Initialize the FaceDetector
face_model_path = 'weights/Alignment_RetinaFace.pth'
face_detector = FaceDetector(model_path=face_model_path, device='cuda')

# Initialize the LandmarkDetector
landmark_model_path =  'weights/Landmark_98.pkl'
landmark_detector = LandmarkDetector(model_path=landmark_model_path, device='cuda', device_ids=[0])

# ===== Configuring Inputs and Outputs ===== #
# Configuring inputs
image_path = '../data/6.jpg'
img_id = image_path.split('/')[-1].split('.')[0]
image_raw = cv2.imread(image_path)

# Configuring outputs
output_dir = "../landmark_layouts"
os.makedirs(output_dir, exist_ok=True)

# Output filename
output_path = os.path.join(
    output_dir,
    f"openface_layout_img{img_id}.png"
)

# ===== Running OpenFace Landmark Detection ===== #
# Detect faces
vis = None
cropped_face, dets = face_detector.get_face(image_path)

# Keep only the highest-confidence detection
best_det = dets[np.argmax(dets[:, 4])]
best_det = best_det[np.newaxis, :]   # Shape becomes (1, 15)

vis = image_raw.copy()

x1, y1, x2, y2 = best_det[0][:4].astype(int)

cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)

cv2.imwrite("det_box.png", vis)

if dets is not None and len(dets) > 0:
    print("Faces detected!")

    # Detect landmarks
    landmarks = landmark_detector.detect_landmarks(image_raw, best_det)

    for idx, (x, y) in enumerate(landmarks[0]):
        print(f"Landmark {idx:2d}: ({x:.1f}, {y:.1f})")

    if landmarks:
        vis = of_utils.visualize_landmarks(image_raw, landmarks)
else:
    print("No faces detected.")

# Save the image
if vis is not None: 
    success = cv2.imwrite(output_path, vis)
else: 
    raise ValueError("Returned layout have no landmarks!")

if success:
    print(f"Saved landmark visualization to: {output_path}")
else:
    print("Failed to save image.")