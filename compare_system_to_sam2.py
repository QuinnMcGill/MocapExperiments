import argparse
import os
import cv2
import numpy as np
import pandas as pd
import sam2_lip_segmentation.sam2_utils as sam2_utils

# ============================================================
# Configuration
# ============================================================

MIN_FRAME_SEPARATION = 100

# ============================================================
# Mocap polygons
# ============================================================

def get_polygon_from_row(row, prefix):
    """
    Extract a polygon from a CSV row.

    Parameters
    ----------
    row : pandas Series
        One row of the polygon CSV.

    prefix : str
        Either "upper" or "lower".

    Returns
    -------
    polygon : (N, 2) numpy array
    """

    x_columns = [
        c for c in row.index
        if c.startswith(f"{prefix}_x_")
    ]

    y_columns = [
        c for c in row.index
        if c.startswith(f"{prefix}_y_")
    ]

    # Sort numerically rather than lexicographically.
    # This matters if there are >9 points.
    x_columns.sort(
        key=lambda x: int(x.split("_")[-1])
    )

    y_columns.sort(
        key=lambda x: int(x.split("_")[-1])
    )

    polygon = np.column_stack(
        [
            row[x_columns].to_numpy(dtype=np.float32),
            row[y_columns].to_numpy(dtype=np.float32),
        ]
    )

    return polygon


def polygon_to_mask(polygon, height, width):
    """
    Rasterize a polygon into a binary mask.
    """

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    polygon_int = np.asarray(
        np.round(polygon),
        dtype=np.int32,
    ).reshape((-1, 1, 2))

    cv2.fillPoly(
        mask,
        [polygon_int],
        (255,),
    )

    return mask.astype(bool)


def get_mocap_masks_from_row(
    row,
    height,
    width,
):
    """
    Extract upper/lower polygons from a CSV row and
    rasterize them into binary masks.
    """

    upper_polygon = get_polygon_from_row(
        row,
        "upper",
    )

    lower_polygon = get_polygon_from_row(
        row,
        "lower",
    )

    upper_mask = polygon_to_mask(
        upper_polygon,
        height,
        width,
    )

    lower_mask = polygon_to_mask(
        lower_polygon,
        height,
        width,
    )

    return (
        upper_mask,
        lower_mask,
        upper_polygon,
        lower_polygon,
    )

# ============================================================
# Metrics
# ============================================================

def calculate_iou(mask_a, mask_b):
    """
    Calculate intersection-over-union between two binary masks.
    """

    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)

    intersection = np.logical_and(
        mask_a,
        mask_b,
    ).sum()

    union = np.logical_or(
        mask_a,
        mask_b,
    ).sum()

    if union == 0:
        return np.nan

    return intersection / union


def calculate_lip_ious(
    sam2_upper,
    sam2_lower,
    mocap_upper,
    mocap_lower,
):
    """
    Calculate upper, lower, and mean lip IoU.
    """

    upper_iou = calculate_iou(
        sam2_upper,
        mocap_upper,
    )

    lower_iou = calculate_iou(
        sam2_lower,
        mocap_lower,
    )

    valid_ious = [
        iou
        for iou in [upper_iou, lower_iou]
        if not np.isnan(iou)
    ]

    if len(valid_ious) == 0:
        mean_iou = np.nan
    else:
        mean_iou = np.mean(valid_ious)

    return upper_iou, lower_iou, mean_iou


# ============================================================
# Frame selection
# ============================================================

def select_high_error_frames(
    results_df,
    n_frames,
    min_frame_separation=100,
):
    """
    Select frames with the highest shape error while enforcing
    a minimum frame separation.

    Frames are considered in order of decreasing error.

    Parameters
    ----------
    results_df : DataFrame
        Must contain "frame" and "error".

    n_frames : int
        Maximum number of frames to select.

    min_frame_separation : int
        Minimum distance between selected frames.

    Returns
    -------
    selected : DataFrame
    """

    sorted_df = results_df.sort_values(
        "error",
        ascending=False,
    )

    selected_indices = []

    for idx in sorted_df.index:

        frame = int(
            sorted_df.loc[idx, "frame"]
        )

        # Check against every already-selected frame.
        too_close = any(
            abs(frame - selected_frame)
            < min_frame_separation
            for selected_frame in selected_indices
        )

        if too_close:
            continue

        selected_indices.append(frame)

        if len(selected_indices) >= n_frames:
            break

    selected = sorted_df[
        sorted_df["frame"].isin(selected_indices)
    ].copy()

    # Return in chronological order.
    selected = selected.sort_values("frame")

    return selected


def select_low_error_frames(
    results_df,
    n_frames,
    min_frame_separation=100,
):
    """
    Select frames with the lowest shape error while enforcing
    a minimum frame separation.

    Frames are considered in order of decreasing error.

    Parameters
    ----------
    results_df : DataFrame
        Must contain "frame" and "error".

    n_frames : int
        Maximum number of frames to select.

    min_frame_separation : int
        Minimum distance between selected frames.

    Returns
    -------
    selected : DataFrame
    """

    sorted_df = results_df.sort_values(
        "error",
        ascending=True,
    )

    selected_indices = []

    for idx in sorted_df.index:

        frame = int(
            sorted_df.loc[idx, "frame"]
        )

        # Check against every already-selected frame.
        too_close = any(
            abs(frame - selected_frame)
            < min_frame_separation
            for selected_frame in selected_indices
        )

        if too_close:
            continue

        selected_indices.append(frame)

        if len(selected_indices) >= n_frames:
            break

    selected = sorted_df[
        sorted_df["frame"].isin(selected_indices)
    ].copy()

    # Return in chronological order.
    selected = selected.sort_values("frame")

    return selected

# ============================================================
# Visualization
# ============================================================

def draw_mask_contour(
    image,
    mask,
    color,
    thickness=2,
):
    """
    Draw the contour of a binary mask.
    """

    mask_uint8 = (
        mask.astype(np.uint8) * 255
    )

    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        image,
        contours,
        -1,
        color,
        thickness,
    )


def overlay_mask(
    image,
    mask,
    color,
    alpha=0.25,
):
    """
    Overlay a binary mask on an image.
    """

    overlay = image.copy()

    overlay[mask] = color

    return cv2.addWeighted(
        overlay,
        alpha,
        image,
        1 - alpha,
        0,
    )


def draw_polygon(
    image,
    polygon,
    color,
    thickness=2,
    point_radius=4
):
    """
    Draw a polygon on an image.
    """

    polygon_int = np.round(
        polygon
    ).astype(np.int32)

    cv2.polylines(
        image,
        [polygon_int],
        isClosed=True,
        color=color,
        thickness=thickness,
    )

    # Draw a dot at each polygon vertex
    for x, y in polygon_int:
        cv2.circle(
            image,
            (int(x), int(y)),
            point_radius,
            color,
            -1,
        )


def create_visualization(
    frame,
    sam2_upper,
    sam2_lower,
    upper_polygon,
    lower_polygon,
    upper_iou,
    lower_iou,
    mean_iou,
    frame_idx,
    system_name,
    crop_box = None
):
    """
    Create a visualization showing SAM2 masks and mocap
    polygons.
    """

    image = frame.copy()

    # --------------------------------------------------------
    # SAM2 mask overlays
    # --------------------------------------------------------

    image = overlay_mask(
        image,
        sam2_upper,
        (255, 0, 0),
        alpha=0.25,
    )

    image = overlay_mask(
        image,
        sam2_lower,
        (0, 0, 255),
        alpha=0.25,
    )

    # --------------------------------------------------------
    # SAM2 contours
    # --------------------------------------------------------

    draw_mask_contour(
        image,
        sam2_upper,
        (255, 0, 0),
        thickness=2,
    )

    draw_mask_contour(
        image,
        sam2_lower,
        (0, 0, 255),
        thickness=2,
    )

    # --------------------------------------------------------
    # Mocap polygons
    # --------------------------------------------------------

    draw_polygon(
        image,
        upper_polygon,
        (0, 255, 255),
        thickness=2,
    )

    draw_polygon(
        image,
        lower_polygon,
        (0, 255, 255),
        thickness=2,
    )

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    error = 1.0 - mean_iou

    cv2.putText(
        image,
        f"{system_name} | Frame {frame_idx}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        image,
        f"Upper IoU: {upper_iou:.3f}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        image,
        f"Lower IoU: {lower_iou:.3f}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        image,
        f"Mean IoU: {mean_iou:.3f}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        image,
        f"Error: {error:.3f}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # --------------------------------------------------------
    # Crop visualization
    # --------------------------------------------------------

    if crop_box is not None:

        x1, y1, x2, y2 = crop_box

        image = image[
            y1:y2,
            x1:x2
        ]

    return image

    return image

def generate_mouth_crop_box(video_path):
    """
    Display the first frame of the video and ask the user to
    draw a bounding box around the mouth region.

    Controls:
        - Left click: select top-left corner
        - Left click again: select bottom-right corner
        - r: redraw the box
        - Enter: confirm the box
        - Esc: cancel

    Returns
    -------
    tuple
        (x1, y1, x2, y2)
    """
    window_name = "Select Mouth Crop"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 1440)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(
            "Could not read the first frame of the video."
        )

    display_frame = frame.copy()

    print("\nPlease draw a bounding box around the mouth region.")
    print("  1. Click the TOP-LEFT corner.")
    print("  2. Click the BOTTOM-RIGHT corner.")
    print("  r = redraw")
    print("  ENTER = confirm")
    print("  ESC = cancel\n")

    points = []

    def mouse_callback(event, x, y, flags, param):

        nonlocal points, display_frame

        if event == cv2.EVENT_LBUTTONDOWN:

            # First click = top-left
            if len(points) == 0:
                points.append((x, y))

            # Second click = bottom-right
            elif len(points) == 1:
                points.append((x, y))

            # Redraw if somehow clicked again
            else:
                points = [(x, y)]

            display_frame = frame.copy()

            if len(points) >= 1:
                cv2.circle(
                    display_frame,
                    points[0],
                    5,
                    (0, 255, 0),
                    -1,
                )

            if len(points) == 2:

                x1, y1 = points[0]
                x2, y2 = points[1]

                cv2.rectangle(
                    display_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(
        window_name,
        mouse_callback,
    )

    while True:

        cv2.imshow(
            window_name,
            display_frame,
        )

        key = cv2.waitKey(20) & 0xFF

        # ----------------------------------------------------
        # Redraw
        # ----------------------------------------------------

        if key == ord("r"):

            points = []
            display_frame = frame.copy()

            print("Redrawing bounding box...")

        # ----------------------------------------------------
        # Confirm
        # ----------------------------------------------------

        elif key == 13:  # Enter

            if len(points) == 2:

                x1, y1 = points[0]
                x2, y2 = points[1]

                # Ensure correct ordering
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])

                # Make sure the box isn't empty
                if x2 > x1 and y2 > y1:

                    cv2.destroyWindow(window_name)

                    print(
                        f"Selected mouth crop: "
                        f"({x1}, {y1}) -> ({x2}, {y2})"
                    )

                    return x1, y1, x2, y2

        # ----------------------------------------------------
        # Cancel
        # ----------------------------------------------------

        elif key == 27:  # ESC

            cv2.destroyWindow(window_name)

            raise RuntimeError(
                "Mouth crop selection cancelled."
            )

# ============================================================
# Main comparison
# ============================================================

def compare_system_to_sam2(
    video_path,
    polygon_csv,
    sam2_npz,
    system_name,
    n_visualizations=10,
    min_frame_separation=100,
    crop_box=None,
):
    """
    Compare one mocap system against SAM2 for an entire video.
    """

    tc_name = video_path.split('/')[-1].split(".mp4")[0]
    high_error_dir = "high_error_frames/" + tc_name 
    low_error_dir = "low_error_frames/" + tc_name 

    os.makedirs(
        high_error_dir,
        exist_ok=True,
    )

    os.makedirs(
        low_error_dir,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load polygon CSV
    # --------------------------------------------------------

    polygon_df = pd.read_csv(
        polygon_csv
    )

    print(
        f"Loaded {len(polygon_df)} polygon rows "
        f"from {polygon_csv}"
    )

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    frame_width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    frame_height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print(
        f"Video: {frame_width}x{frame_height}, "
        f"{frame_count} frames"
    )

    # --------------------------------------------------------
    # Load SAM2 sparse masks
    # --------------------------------------------------------

    (
        upper_indices,
        lower_indices,
        upper_offsets,
        lower_offsets,
    ) = sam2_utils.load_raw_sam2_masks(
        sam2_npz
    )

    # --------------------------------------------------------
    # Compare every frame
    # --------------------------------------------------------

    results = []

    frame_idx = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # ----------------------------------------------------
        # Make sure a corresponding CSV row exists
        # ----------------------------------------------------

        matching_rows = polygon_df[
            polygon_df["frame"] == frame_idx
        ]

        if len(matching_rows) == 0:
            frame_idx += 1
            continue

        row = matching_rows.iloc[0]

        # ----------------------------------------------------
        # Reconstruct SAM2 masks
        # ----------------------------------------------------

        sam2_upper, sam2_lower = sam2_utils.reconstruct_sam2_masks(
            frame_idx,
            upper_indices,
            lower_indices,
            upper_offsets,
            lower_offsets,
            frame_height,
            frame_width,
        )

        # ----------------------------------------------------
        # Reconstruct mocap masks
        # ----------------------------------------------------

        (
            mocap_upper,
            mocap_lower,
            upper_polygon,
            lower_polygon,
        ) = get_mocap_masks_from_row(
            row,
            frame_height,
            frame_width,
        )

        # ----------------------------------------------------
        # Calculate IoU
        # ----------------------------------------------------

        (
            upper_iou,
            lower_iou,
            mean_iou,
        ) = calculate_lip_ious(
            sam2_upper,
            sam2_lower,
            mocap_upper,
            mocap_lower,
        )

        if np.isnan(mean_iou):
            frame_idx += 1
            continue

        error = 1.0 - mean_iou

        results.append(
            {
                "frame": frame_idx,
                "timestamp": row["timestamp"],
                "upper_iou": upper_iou,
                "lower_iou": lower_iou,
                "mean_iou": mean_iou,
                "error": error,
            }
        )

        frame_idx += 1

    cap.release()

    # --------------------------------------------------------
    # Save frame-by-frame metrics
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    metrics_path = os.path.join(
        high_error_dir,
        f"{system_name}_sam2_iou.csv",
    )

    results_df.to_csv(
        metrics_path,
        index=False,
    )

    print(
        f"Saved metrics to {metrics_path}"
    )

    # --------------------------------------------------------
    # Select highest-error frames
    # --------------------------------------------------------

    print("\nAssessing frames with the highest IoU error...")

    high_error_df = select_high_error_frames(
        results_df,
        n_visualizations,
        min_frame_separation,
    )

    high_error_path = os.path.join(
        high_error_dir,
        f"{system_name}_highest_error_frames.csv",
    )

    high_error_df.to_csv(
        high_error_path,
        index=False,
    )

    print(
        f"Selected frames for visualization:"
    )

    print(
        high_error_df[
            [
                "frame",
                "upper_iou",
                "lower_iou",
                "mean_iou",
                "error",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Select lowest-error frames
    # --------------------------------------------------------

    print("\nAssessing frames with the lowest IoU error...")

    low_error_df = select_low_error_frames(
        results_df,
        n_visualizations,
        min_frame_separation,
    )

    low_error_path = os.path.join(
        low_error_dir,
        f"{system_name}_lowest_error_frames.csv",
    )

    low_error_df.to_csv(
        low_error_path,
        index=False,
    )

    print(
        f"Selected frames for visualization:"
    )

    print(
        low_error_df[
            [
                "frame",
                "upper_iou",
                "lower_iou",
                "mean_iou",
                "error",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Reopen video for visualization
    # --------------------------------------------------------

    print("\nCreating visuals for highest error frames...")

    cap = cv2.VideoCapture(
        video_path
    )

    for _, result in high_error_df.iterrows():

        target_frame = int(
            result["frame"]
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            target_frame,
        )

        ret, frame = cap.read()

        if not ret:
            print(
                f"Could not read frame {target_frame}"
            )
            continue

        # ----------------------------------------------------
        # Retrieve polygon row
        # ----------------------------------------------------

        matching_rows = polygon_df[
            polygon_df["frame"] == target_frame
        ]

        if len(matching_rows) == 0:
            continue

        row = matching_rows.iloc[0]

        # ----------------------------------------------------
        # Reconstruct masks
        # ----------------------------------------------------

        sam2_upper, sam2_lower = sam2_utils.reconstruct_sam2_masks(
            target_frame,
            upper_indices,
            lower_indices,
            upper_offsets,
            lower_offsets,
            frame_height,
            frame_width,
        )

        (
            _,
            _,
            upper_polygon,
            lower_polygon,
        ) = get_mocap_masks_from_row(
            row,
            frame_height,
            frame_width,
        )

        # ----------------------------------------------------
        # Visualization
        # ----------------------------------------------------

        image = create_visualization(
            frame,
            sam2_upper,
            sam2_lower,
            upper_polygon,
            lower_polygon,
            result["upper_iou"],
            result["lower_iou"],
            result["mean_iou"],
            target_frame,
            system_name,
            crop_box,
        )

        output_path = os.path.join(
            high_error_dir,
            f"{system_name}_frame_{target_frame:06d}.png",
        )

        cv2.imwrite(
            output_path,
            image,
        )

        print(
            f"Saved {output_path}"
        )

    print("\nCreating visuals for lowest error frames...")

    for _, result in low_error_df.iterrows():

        target_frame = int(
            result["frame"]
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            target_frame,
        )

        ret, frame = cap.read()

        if not ret:
            print(
                f"Could not read frame {target_frame}"
            )
            continue

        # ----------------------------------------------------
        # Retrieve polygon row
        # ----------------------------------------------------

        matching_rows = polygon_df[
            polygon_df["frame"] == target_frame
        ]

        if len(matching_rows) == 0:
            continue

        row = matching_rows.iloc[0]

        # ----------------------------------------------------
        # Reconstruct masks
        # ----------------------------------------------------

        sam2_upper, sam2_lower = sam2_utils.reconstruct_sam2_masks(
            target_frame,
            upper_indices,
            lower_indices,
            upper_offsets,
            lower_offsets,
            frame_height,
            frame_width,
        )

        (
            _,
            _,
            upper_polygon,
            lower_polygon,
        ) = get_mocap_masks_from_row(
            row,
            frame_height,
            frame_width,
        )

        # ----------------------------------------------------
        # Visualization
        # ----------------------------------------------------

        image = create_visualization(
            frame,
            sam2_upper,
            sam2_lower,
            upper_polygon,
            lower_polygon,
            result["upper_iou"],
            result["lower_iou"],
            result["mean_iou"],
            target_frame,
            system_name,
            crop_box,
        )

        output_path = os.path.join(
            low_error_dir,
            f"{system_name}_frame_{target_frame:06d}.png",
        )

        cv2.imwrite(
            output_path,
            image,
        )

        print(
            f"Saved {output_path}"
        )

    cap.release()

# ============================================================
# Command line interface
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compare mocap lip polygons against "
            "SAM2 lip segmentation masks."
        )
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Input video",
    )

    parser.add_argument(
        "--polygons",
        required=True,
        help="Mocap polygon CSV",
    )

    parser.add_argument(
        "--sam2",
        required=True,
        help="SAM2 sparse mask NPZ",
    )

    parser.add_argument(
        "--system",
        required=True,
        help="Mocap system name, e.g. mediapipe or openface",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of high-error frames to visualize",
    )

    parser.add_argument(
        "--min-separation",
        type=int,
        default=100,
        help="Minimum frame separation between visualizations",
    )

    args = parser.parse_args()

    crop_box = generate_mouth_crop_box(args.video)

    compare_system_to_sam2(
        video_path=args.video,
        polygon_csv=args.polygons,
        sam2_npz=args.sam2,
        system_name=args.system,
        n_visualizations=args.n,
        min_frame_separation=args.min_separation,
        crop_box=crop_box
    )


if __name__ == "__main__":
    main()