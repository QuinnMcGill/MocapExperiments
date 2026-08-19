import cv2
import argparse

def get_hsv_at_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        frame, hsv_frame = param

        # OpenCV uses H: 0-179, S: 0-255, V: 0-255
        h, s, v = hsv_frame[y, x]

        print(f"Clicked ({x}, {y}) -> HSV: ({h}, {s}, {v})")


parser = argparse.ArgumentParser()
parser.add_argument("--v", required=True, help="Path to video")
parser.add_argument("--f", type=int, required=True, help="Frame index")
args = parser.parse_args()


# Open video
cap = cv2.VideoCapture(args.v)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {args.v}")


# Jump to requested frame
cap.set(cv2.CAP_PROP_POS_FRAMES, args.f)

ret, frame = cap.read()

if not ret:
    cap.release()
    raise RuntimeError(f"Could not read frame {args.f}")


# Convert BGR -> HSV
hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


# Create window
window_name = f"Frame {args.f}"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 640, 1440)

# Pass both the original frame and HSV frame to callback
cv2.setMouseCallback(
    window_name,
    get_hsv_at_click,
    (frame, hsv_frame)
)


print(f"Displaying frame {args.f}")
print("Click anywhere to print HSV values.")
print("Press ESC to close.")

while True:
    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break


cv2.destroyAllWindows()
cap.release()