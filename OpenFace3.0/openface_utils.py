import cv2
import numpy as np

def visualize_landmarks(image, landmarks, radius=6, color=(0, 255, 0)):
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

    vis = image.copy()

    for face_landmarks in landmarks:

        # Convert tensor to numpy if needed
        if hasattr(face_landmarks, "cpu"):
            face_landmarks = face_landmarks.cpu().numpy()

        face_landmarks = np.asarray(face_landmarks)

        for x, y in face_landmarks:
            cv2.circle(
                vis,
                (int(round(x)), int(round(y))),
                radius,
                color,
                -1
            )

    return vis


def visualize_landmarks_with_ids(image, landmarks):
    vis = image.copy()

    for face_landmarks in landmarks:

        if hasattr(face_landmarks, "cpu"):
            face_landmarks = face_landmarks.cpu().numpy()

        face_landmarks = np.asarray(face_landmarks)

        for idx, (x, y) in enumerate(face_landmarks):
            x, y = int(round(x)), int(round(y))

            cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
            cv2.putText(
                vis,
                str(idx),
                (x + 3, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

    cv2.imshow("Facial Landmarks", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()