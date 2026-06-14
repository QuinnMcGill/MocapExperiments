import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import cv2
import time
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

import mp_helpers as mph

# Initialize Face Mesh module
base_options = python.BaseOptions(model_asset_path="MyMediaPipe/models/face_landmarker_v2_with_blendshapes.task")
options = vision.FaceLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO, output_face_blendshapes=True, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# Capture video from webcam
cap = cv2.VideoCapture(0)

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

    # Get the timestamp before processing
    timestamp_ms = int(time.time() * 1000)

    # Process the frame and detect face landmarks
    results = detector.detect_for_video(mp_image, timestamp_ms)

    # Draw face landmarks on the image
    #drawing_utils.draw_landmarks(frame, face_landmarks, drawing_styles.get_default_face_mesh_tesselation_style())
    annotated_image = mph.draw_landmarks_on_image(mp_image.numpy_view(), results)

    # Display the resulting frame
    annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Face Detection", annotated_bgr)

    # ==== Flow control ==== #
    key = cv2.waitKey(5) & 0xFF
    if key == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()