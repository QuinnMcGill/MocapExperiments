import cv2
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import mediapipe as mp

# Import MediaPipe modules
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

# Import custom helper functions for MediaPipe
import mp_helpers as mph

# ===== Configuring MediaPipe Model ===== #
# Load the .task model and set up the options
base_options = python.BaseOptions(model_asset_path="MediaPipe/models/face_landmarker_v2_with_blendshapes.task")
options = vision.FaceLandmarkerOptions(base_options=base_options, 
                                       running_mode=vision.RunningMode.IMAGE, 
                                       output_face_blendshapes=False, 
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# ===== Configuring Inputs and Outputs ===== #
# Configuring inputs
data_dir = "/home/quinnm/repo/MocapExperiments/data/"
for filename in os.listdir(data_dir):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue  # Skip non-image files
    img_path = os.path.join(data_dir, filename)
    img_id = img_path.split('/')[-1].split('.')[0]
    img = mp.Image.create_from_file(img_path)

    # Configuring outputs
    output_dir = "/home/quinnm/repo/MocapExperiments/landmark_layouts"
    os.makedirs(output_dir, exist_ok=True)

    # Output filename
    output_path = os.path.join(
        output_dir,
        f"mediapipe_layout_img{img_id}.png"
    )

    # ===== Running Mediapipe Landmark Detection ===== #
    detection_result = detector.detect(img)
    landmarks = detection_result.face_landmarks[0]  # Only consider the first detected face for visualization

    annotated_image = mph.visualize_landmarks(cv2.imread(img_path), landmarks)
    annotated_image = mph.draw_mediapipe_lip_contours(annotated_image, detection_result)

    # Save the image
    if annotated_image is not None: 
        success = cv2.imwrite(output_path, annotated_image)
        print("Created layout file in: ", output_path)
    else: 
        raise ValueError("Returned layout have no landmarks!")