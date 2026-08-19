import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import numpy as np
import matplotlib.pyplot as plt

# For Visualization
RED_COLOR = (255, 0, 0)
BLUE_COLOUR = (0, 0, 255)
BLACK_COLOR = (0, 0, 0)
WHITE_COLOR = (255, 255, 255)

# Medapipe landmark indices along lip contours
MEDIAPIPE_MOUTH_LAYOUT = {
    "top_outer": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],           # Left outer corner to right outer corner (includes corners)
    "top_inner": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],          # Left inner corner to left inner corner (includes corners)
    "bottom_inner": [308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78],                # Right inner corner to left inner corner (excludes corners)
    "bottom_outer": [291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]                # Right outer corner to left outer corner (excludes corners)
}

MEDIAPIPE_LIP_CONTOUR_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                                 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
                                 324, 318, 402, 317, 14, 87, 178, 88, 95,
                                 375, 321, 405, 314, 17, 84, 181, 91, 146]

def draw_face_mesh(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style())
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style())
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())

    return annotated_image

def draw_mediapipe_lip_contour_landmarks(rgb_image, detection_result, circle_radius=5, landmark_color=RED_COLOR, add_indices=False):
    face_landmarks_list = detection_result.face_landmarks
    face_landmarks = face_landmarks_list[0]  # Assuming only one face is detected for simplicity
    annotated_image = np.copy(rgb_image)

    h, w = rgb_image.shape[:2]

    for idx in MEDIAPIPE_LIP_CONTOUR_INDICES:

        lm = face_landmarks[idx]

        px = int(lm.x * w)
        py = int(lm.y * h)

        cv2.circle(annotated_image, (px, py), circle_radius, landmark_color, -1)

        if add_indices:
            cv2.putText(
                annotated_image,
                str(idx),
                (px + 3, py - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                BLACK_COLOR,
                1
            )
        
    return annotated_image

def draw_mediapipe_lip_contours(rgb_image, detection_result, contour_color=WHITE_COLOR):

    face_landmarks_list = detection_result.face_landmarks
    face_landmarks = face_landmarks_list[0]  # Assuming only one face is detected for simplicity
    annotated_image = np.copy(rgb_image)
    
    # MediaPipe lip connections
    lip_connections = (
        vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS
    )

    for connection in lip_connections:

        start_idx = connection.start
        end_idx = connection.end

        start_landmark = face_landmarks[start_idx]
        end_landmark = face_landmarks[end_idx]

        # Convert normalized coordinates to pixels
        start_x = int(
            start_landmark.x * annotated_image.shape[1]
        )

        start_y = int(
            start_landmark.y * annotated_image.shape[0]
        )

        end_x = int(
            end_landmark.x * annotated_image.shape[1]
        )

        end_y = int(
            end_landmark.y * annotated_image.shape[0]
        )

        cv2.line(
            annotated_image,
            (start_x, start_y),
            (end_x, end_y),
            contour_color,
            thickness=2
        )

    return annotated_image

def visualize_landmarks(image, landmarks, radius=6, color=(0, 255, 0), draw_indices=False):
    """
    Display facial landmarks overlaid on an image.

    Parameters
    ----------
    image : np.ndarray
        Original BGR image loaded by cv2.imread().
    landmarks : list
        Output from landmark_detector.detect_landmarks().
        Expected format:
            landmarks[face_idx] -> (N,2) array of (x,y) coordinates.
    radius : int
        Radius of landmark points.
    color : tuple
        BGR color for landmark points.

    Returns
    -------
    vis : np.ndarray
        Copy of image with overlayed landmark locations.
    """

    h, w = image.shape[:2]

    vis = image.copy()

    for lm in landmarks:

        px = int(lm.x * w)
        py = int(lm.y * h)

        cv2.circle(
            vis,
            (px, py),
            3,
            color,  # Red
            -1
        )

    return vis

# def draw_mediapipe_lip_contours(rgb_image, detection_result):
#     face_landmarks_list = detection_result.face_landmarks
#     annotated_image = np.copy(rgb_image)

#     landmark_drawing_spec = drawing_utils.DrawingSpec(
#         color=RED_COLOR, thickness=5
#     )

#     contours_drawing_spec = drawing_utils.DrawingSpec(
#         color=BLACK_COLOR, thickness=5
#     )

#     # Loop through the detected faces to visualize.
#     for idx in range(len(face_landmarks_list)):
#         face_landmarks = face_landmarks_list[idx]
    
#         drawing_utils.draw_landmarks(
#                 image=annotated_image,
#                 landmark_list=face_landmarks,
#                 connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS,
#                 landmark_drawing_spec=landmark_drawing_spec,
#                 connection_drawing_spec=contours_drawing_spec)
        
#     return annotated_image

# ====== Saving Mediapipe Results ====== #

def create_mediapipe_lip_polygon_row(
    detection_result,
    frame_idx,
    timestamp,
    frame_width,
    frame_height
):
    """
    Create a CSV row containing upper- and lower-lip polygons
    from MediaPipe Face Landmarker landmarks.

    Parameters
    ----------
    face_landmarks :
        MediaPipe face landmarks detection result.

    frame :
        Video frame index.

    timestamp :
        Frame timestamp.

    face_id :
        MediaPipe face index.

    frame_width :
        Width of the original video frame in pixels.

    frame_height :
        Height of the original video frame in pixels.

    Returns
    -------
    dict
        Dictionary containing frame metadata and polygon
        coordinates in pixel space.
    """

    # ---------------------------------------------------------
    # Convert all normalized MediaPipe landmarks to pixel coords
    # ---------------------------------------------------------

    face_landmarks_list = detection_result.face_landmarks
    face_landmarks = face_landmarks_list[0]  # Assuming only one face is detected for simplicity

    landmarks = np.array([
        [
            landmark.x * frame_width,
            landmark.y * frame_height
        ]
        for landmark in face_landmarks
    ], dtype=np.float32)

    # ---------------------------------------------------------
    # Extract the four lip contours
    # ---------------------------------------------------------

    top_outer = landmarks[MEDIAPIPE_MOUTH_LAYOUT["top_outer"]]

    top_inner = landmarks[MEDIAPIPE_MOUTH_LAYOUT["top_inner"]]

    bottom_inner = landmarks[MEDIAPIPE_MOUTH_LAYOUT["bottom_inner"]]

    bottom_outer = landmarks[MEDIAPIPE_MOUTH_LAYOUT["bottom_outer"]]


    # ---------------------------------------------------------
    # Construct upper lip polygon
    #
    # top_outer:
    #     left -> right
    #
    # top_inner:
    #     left -> right
    #
    # Reverse top_inner so the polygon travels continuously
    # around the upper lip.
    # ---------------------------------------------------------

    upper_polygon = np.concatenate(
        [
            top_outer,
            top_inner[::-1]
        ],
        axis=0
    )

    # ---------------------------------------------------------
    # Construct lower lip polygon
    #
    # bottom_inner:
    #     right -> left
    #
    # bottom_outer:
    #     right -> left
    #
    # Reverse bottom_outer so that we travel continuously
    # around the lower lip.
    # ---------------------------------------------------------

    lower_polygon = np.concatenate(
        [
            bottom_inner,
            bottom_outer[::-1]
        ],
        axis=0
    )

    # ---------------------------------------------------------
    # Build CSV row
    # ---------------------------------------------------------

    row = {
        "frame": frame_idx,
        "timestamp": timestamp
    }

    # ---------------------------------------------------------
    # Upper polygon coordinates
    # ---------------------------------------------------------

    for i, (x, y) in enumerate(upper_polygon):

        row[f"upper_x_{i}"] = x
        row[f"upper_y_{i}"] = y

    # ---------------------------------------------------------
    # Lower polygon coordinates
    # ---------------------------------------------------------

    for i, (x, y) in enumerate(lower_polygon):

        row[f"lower_x_{i}"] = x
        row[f"lower_y_{i}"] = y

    return row