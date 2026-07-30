# video_detection.py

import cv2
import numpy as np
import time


def compare_white_pixels(
    current_frame, paused_gray, threshold=70, white_threshold=0.85
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
    y1, y2 = 40, 120
    x1, x2 = 1764, 1853
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


def process_video(video_path, pause_image):
    """
    Process a video and detect events based on comparison of frames to a paused image.

    Args:
    - video_path (str): Path to the video file.
    - pause_image (ndarray): The paused image to compare each frame to.

    Returns:
    - list: A list of detection results (True/False for each frame).
    """

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open the video file.")
        return []  # Exit early if the video couldn't be opened

    # List to store the results (True/False for each frame)
    frame_results = []

    # Initialize a frame counter
    frame_counter = 0
    # print("Starting detection process!")
    t0 = time.time()
    paused_gray = cv2.cvtColor(pause_image, cv2.COLOR_BGR2GRAY)
    while cap.isOpened():
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if not ret:
            # print("Error: Failed to read frame or end of video reached.")
            break  # End of video or error

        # Perform image recognition (compare white pixels or other analysis)
        is_detected = compare_white_pixels(frame, paused_gray)

        frame_results.append(is_detected)

        # Increment the frame counter
        frame_counter += 1

        # Print progress every 1800 frames
        if frame_counter % 1800 == 0:
            print(f"Processed {frame_counter/1800} mins...")

        # # Exit the loop if 'q' is pressed
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break
    t1 = time.time()
    print(
        f"Detection process completed in {t1-t0:.3f} seconds for {frame_counter} frames."
    )
    # Release the video capture and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    return frame_results


def detect_event_changes(detection_results):
    """
    Detect event changes based on detection results (True/False list).

    Args:
    - detection_results (list): A list of detection results (True/False for each frame).

    Returns:
    - list: A list of indices where events occurred (when detection results change).
    """

    # Initialize the list with 0 by default
    event_indices = [0]

    # Keep track of the last frame's result
    last_result = detection_results[0]  # Start with the first frame result

    # Iterate through the detection results starting from the second frame
    for i in range(1, len(detection_results)):
        # If the current frame's result differs from the last, record the event
        if detection_results[i] != last_result:
            event_indices.append(i)  # Add the index of the event to the list
            last_result = detection_results[
                i
            ]  # Update the last_result to the current frame's result

    # Add the size of the original list of detection results to the list
    event_indices.append(len(detection_results))

    return event_indices
