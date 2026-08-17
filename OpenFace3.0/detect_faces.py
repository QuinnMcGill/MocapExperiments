import cv2
from openface.face_detection import FaceDetector    #type:ignore

# Initialize the FaceDetector
model_path = 'weights/Alignment_RetinaFace.pth'
detector = FaceDetector(model_path=model_path, device='cuda')

# Path to the input image
image_path = '../data/6.jpg'
img_id = image_path.split('/')[-1].split('.')[0]

# Detect and extract the face
cropped_face, dets = detector.get_face(image_path)

if cropped_face is not None:
    print("Face detected!")
    print(f"Detection results: {dets}")
    
    # Save the cropped face as an image
    output_path = f'detected_faces/{img_id}.jpg'
    cv2.imwrite(output_path, cropped_face)
    print(f"Detected face saved to: {output_path}")
else:
    print("No face detected.")