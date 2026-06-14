import cv2

from openface.face_detection import FaceDetector
from openface.landmark_detection import LandmarkDetector

# Path to submodules
submodule_path = '../external/OpenFace'

# Initialize the FaceDetector
face_model_path = submodule_path + '/weights/Alignment_RetinaFace.pth'
face_detector = FaceDetector(model_path=face_model_path, device='cuda')

# Initialize the LandmarkDetector
landmark_model_path = submodule_path + '/weights/Landmark_98.pkl'
landmark_detector = LandmarkDetector(model_path=landmark_model_path, device='cuda')

# Path to the input image
image_path = 'path/to/input_image.jpg'
image_raw = cv2.imread(image_path)

# Detect faces
cropped_face, dets = face_detector.get_face(image_path)

if dets is not None and len(dets) > 0:
    print("Faces detected!")

    # Detect landmarks
    landmarks = landmark_detector.detect_landmarks(image_raw, dets)
    if landmarks:
        for i, landmark in enumerate(landmarks):
            print(f"Landmarks for face {i}: {landmark}")
else:
    print("No faces detected.")