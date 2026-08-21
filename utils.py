import cv2 
import numpy as np

#  ==== Testcase 13 ==== #

# Green Mask Values [min, max]
TC13_GREEN_H = [50, 130]
TC13_GREEN_S = [20, 100]
TC13_GREEN_V = [20, 100]

# Red Mask 1 Values [min, max]
TC13_RED1_H = [0, 14]
TC13_RED1_S = [70, 100]
TC13_RED1_V = [22, 100]

# Red Mask 2 Values [min, max]
TC13_RED2_H = [330, 355]
TC13_RED2_S = [60, 100]
TC13_RED2_V = [50, 100]

# ==== Testcase 14 ==== #

# Green Mask 1 Values [min, max]
TC14_GREEN1_H = [35, 130]
TC14_GREEN1_S = [20, 100]
TC14_GREEN1_V = [20, 100]

# Green Mask 2 Values [min, max]
TC14_GREEN2_H = [30, 35]
TC14_GREEN2_S = [35, 40]
TC14_GREEN2_V = [30, 40]

# Red Mask 1 Values [min, max]
TC14_RED1_H = [0, 14]
TC14_RED1_S = [70, 100]
TC14_RED1_V = [22, 100]

# Red Mask 2 Values [min, max]
TC14_RED2_H = [330, 355]
TC14_RED2_S = [60, 100]
TC14_RED2_V = [42, 100]

def get_hsv_vals(tc_name):
    green1_mask = None
    green2_mask = None
    red_mask1 = None
    red_mask2 = None

    if tc_name == "tc13":
        green1_mask = (TC13_GREEN_H, TC13_GREEN_S, TC13_GREEN_V)
        red_mask1 = (TC13_RED1_H, TC13_RED1_S, TC13_RED1_V)
        red_mask2 = (TC13_RED2_H, TC13_RED2_S, TC13_RED2_V)
    elif tc_name == "tc14":
        green1_mask = (TC14_GREEN1_H, TC14_GREEN1_S, TC14_GREEN1_V)
        green2_mask = (TC14_GREEN2_H, TC14_GREEN2_S, TC14_GREEN2_V)
        red_mask1 = (TC14_RED1_H, TC14_RED1_S, TC14_RED1_V)
        red_mask2 = (TC14_RED2_H, TC14_RED2_S, TC14_RED2_V)
    else:
        raise SystemExit("Failed to recognize testcase name")

    return green1_mask, green2_mask, red_mask1, red_mask2

def get_masks(tc_name, hsv):
    green1_hsv, green2_hsv, red1_hsv, red2_hsv = get_hsv_vals(tc_name)
    [G1_h1, G1_h2], [G1_s1, G1_s2], [G1_v1, G1_v2] = green1_hsv

    if green2_hsv is not None: 
        [G2_h1, G2_h2], [G2_s1, G2_s2], [G2_v1, G2_v2] = green2_hsv
        green_2_start = np.array([round(G2_h1/355*180), round(G2_s1/100*255), round(G2_v1/100*255)])
        green_2_end = np.array([round(G2_h2/355*180), round(G2_s2/100*255), round(G2_v2/100*255)])
    else:
        green_2_start = None
        green_2_end = None

    [R1_h1, R1_h2], [R1_s1, R1_s2], [R1_v1, R1_v2] = red1_hsv
    [R2_h1, R2_h2], [R2_s1, R2_s2], [R2_v1, R2_v2] = red2_hsv

    # Define Mask in OpenCV colour space (H: 0-180, S: 0-255, V: 0-255)
    green_1_start = np.array([round(G1_h1/355*180), round(G1_s1/100*255), round(G1_v1/100*255)])
    green_1_end = np.array([round(G1_h2/355*180), round(G1_s2/100*255), round(G1_v2/100*255)])

    red_1_start = np.array([round(R1_h1/355*180), round(R1_s1/100*255), round(R1_v1/100*255)])
    red_1_end = np.array([round(R1_h2/355*180), round(R1_s2/100*255), round(R1_v2/100*255)])

    red_2_start = np.array([round(R2_h1/355*180), round(R2_s1/100*255), round(R2_v1/100*255)])
    red_2_end = np.array([round(R2_h2/355*180), round(R2_s2/100*255), round(R2_v2/100*255)])

    green_mask_1 = cv2.inRange(
        hsv,
        green_1_start,
        green_1_end
    ) 

    if green_2_start is not None and green_2_end is not None:
        green_mask_2 = cv2.inRange(
            hsv,
            green_2_start,
            green_2_end
        ) 
    else:
        green_mask_2 = None

    red_mask_1 = cv2.inRange(
        hsv, 
        red_1_start, 
        red_1_end
    )

    red_mask_2 = cv2.inRange(
        hsv, 
        red_2_start, 
        red_2_end
    )

    if green_mask_2 is not None:
        green_mask = cv2.bitwise_or(green_mask_1, green_mask_2)
    else:
        green_mask = green_mask_1

    red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

    return green_mask, red_mask

def largest_connected_component(mask):
    """
    Returns a binary mask containing only the largest connected component.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (0 and 255).

    Returns
    -------
    largest_mask : np.ndarray
        Binary mask with only the largest connected component.
    """

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    # No foreground pixels.
    if num_labels <= 1:
        return np.zeros_like(mask)

    # Ignore background (label 0).
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    largest_mask = np.zeros_like(mask)
    largest_mask[labels == largest_label] = 255

    return largest_mask

def close_mask_holes(mask, kernel_size=8, iterations=1):
    """
    Fill small holes inside a binary mask using morphological closing.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask with values 0 and 255.
    kernel_size : int
        Size of the elliptical structuring element.
    iterations : int
        Number of closing iterations.

    Returns
    -------
    closed_mask : np.ndarray
        Binary mask with holes filled.
    """

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    closed_mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=iterations
    )

    return closed_mask