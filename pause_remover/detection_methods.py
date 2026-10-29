import cv2
import numpy as np
import time
from skimage.metrics import structural_similarity as ssim

roi_coords = {"pc": (1790, 1860, 45, 105), "emu": (1770, 1850, 40, 120)}


def get_ssim_score(test_img, paused_img, pc_client=True):
    x1, x2, y1, y2 = roi_coords["pc"] if pc_client else roi_coords["emu"]
    paused_roi = paused_img[y1:y2, x1:x2]
    test_roi = test_img[y1:y2, x1:x2]
    if len(test_img.shape) == 3 and test_img.shape[2] == 3:
        test_roi = cv2.cvtColor(test_roi, cv2.COLOR_BGR2GRAY)
    if len(paused_img.shape) == 3 and paused_img.shape[2] == 3:
        paused_roi = cv2.cvtColor(paused_roi, cv2.COLOR_BGR2GRAY)

    assert (
        paused_roi.shape == test_roi.shape
    ), f"Reference and test ROIs do not match in shape: {paused_roi.shape} vs {test_roi.shape}"
    return ssim(test_roi, paused_roi)


def compare_ssim(test_img, paused_img, threshold=0.5, pc_client=True) -> bool:
    score = get_ssim_score(test_img, paused_img, pc_client)
    return score > threshold


def get_canny_score(img, ref_img, pc_client=True):
    x1, x2, y1, y2 = roi_coords["emu"]

    img_roi = img[y1:y2, x1:x2]
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    ref_roi = ref_img[y1:y2, x1:x2]
    if len(ref_img.shape) == 3 and ref_img.shape[2] == 3:
        ref_roi = cv2.cvtColor(ref_roi, cv2.COLOR_BGR2GRAY)
    ref_canny = cv2.Canny(ref_roi, 50, 100)
    img_canny = cv2.Canny(img_roi, 50, 100)
    res = cv2.matchTemplate(
        img_canny,
        ref_canny,
        cv2.TM_CCOEFF_NORMED,
    )
    return cv2.minMaxLoc(res)[1]


def compare_canny(img, ref_img, threshold=0.4, pc_client=True) -> bool:
    score = get_canny_score(img, ref_img, pc_client)
    return score > threshold


def compare_white_pixels(
    current_frame,
    paused_gray,
    threshold=70,
    white_threshold=0.85,
    pc_client=False,
):
    """
    Compare two images based on their white pixels.

    Args:
    - current_frame (ndarray): The current frame (as a raw cv2 object).
    - paused_image (ndarray): The paused image (as a raw cv2 object).
    - threshold (int): Pixel intensity threshold to consider a pixel as "white". Default is 200.
    - white_threshold (float): Percentage of white pixels in the paused image that must correspond
                               to white pixels in the current frame. Default is 0.9 (90%).

    Returns:
    - bool: True if 90% of the white pixels in the paused image correspond to white pixels in the current frame.
    """
    # memo
    x1, x2, y1, y2 = roi_coords["pc"] if pc_client else roi_coords["emu"]
    if pc_client:
        white_threshold = 0.78
    # Ensure both images are loaded
    if current_frame is None or paused_gray is None:
        print("Error: One or both images are None")
        return False

    # Convert images to grayscale
    # TODO edit these margins to fit other resolutions
    roi_current = cv2.cvtColor(current_frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    roi_paused = paused_gray[y1:y2, x1:x2]
    # roi = current_gray
    # roi_paused = paused_gray

    # Create masks for white pixels (those with pixel intensity above the threshold)
    # Count the number of white pixels in the paused image
    current_white_mask = roi_current > threshold
    paused_white_mask = roi_paused > threshold
    paused_white_pixels = np.count_nonzero(paused_white_mask)
    # No white pixels in paused image, cannot compare
    if paused_white_pixels == 0:
        raise Exception("Please check the paused image. No white pixels found.")

    # Count the number of corresponding white pixels in the current frame
    matching_white_pixels = np.count_nonzero(current_white_mask & paused_white_mask)
    # Calculate the percentage of white pixels in the paused image that match in the current frame
    match_percentage = matching_white_pixels / paused_white_pixels
    # Return True if 90% or more of the white pixels in the paused image match the current frame
    return match_percentage >= white_threshold
