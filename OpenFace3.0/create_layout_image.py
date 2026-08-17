import ast
import cv2
import pandas as pd

# ---------------- Configuration ---------------- #
for i in range(6):
    img_id = i + 1
    csv_path = f"/home/quinnm/repo/MocapExperiments/OpenFace3.0/csv_files/face_analysis_{img_id}.csv"      # Path to the OpenFace CSV
    row_idx = 0                   # First detected face

    draw_indices = False           # Draw landmark numbers
    point_radius = 5
    font_scale = 0.35

    # ---------------- Load CSV ---------------- #
    df = pd.read_csv(csv_path)

    row = df.iloc[row_idx]

    image_path = row["image_path"]
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(image_path)

    # ---------------- Parse landmarks ---------------- #
    landmarks = ast.literal_eval(row["landmarks"])

    # landmarks is now a list of [x, y]

    # ---------------- Draw landmarks ---------------- #
    for i, (x, y) in enumerate(landmarks):

        x = int(round(x))
        y = int(round(y))

        cv2.circle(image, (x, y), point_radius, (0, 255, 0), -1)

        if draw_indices:
            cv2.putText(
                image,
                str(i),
                (x + 3, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    # ---------------- Draw face box ---------------- #
    bbox = ast.literal_eval(row["face_detection"])

    x1, y1, x2, y2 = map(int, bbox[:4])

    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # ---------------- Save ---------------- #
    output_path = f"/home/quinnm/repo/MocapExperiments/landmark_layouts/openface_layout_img{img_id}.png"

    success = cv2.imwrite(output_path, image)

    if success:
        print(f"Saved visualization to: {output_path}")
    else:
        print("Failed to save visualization.")