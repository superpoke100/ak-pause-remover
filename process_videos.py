import os
import cv2
import time
import ffmpeg
from pause_detect_utils import (
    compare_white_pixels,
)


def split_and_rewrite_video(
    input_folder="TODO",
    output_parent_folder="output_videos",
    nvidia_gpu=True,
    pc_client=False,
):
    if pc_client:
        pause_image = cv2.imread("pc_pause.png")
    else:
        pause_image = cv2.imread("cnpause.png")
    if pause_image is None:
        raise Exception("Error: Could not load the pause image.")
    # Get all video files (mkv, mov) in the input folder
    video_files = [
        f for f in os.listdir(input_folder) if f.endswith((".mkv", ".mov", "mp4"))
    ]

    # Process each video file
    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)

        # Get the base name (without extension) for the output folder
        base_name = os.path.splitext(video_file)[0]
        output_folder = os.path.join(output_parent_folder, base_name)

        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created folder: {output_folder}")
        else:
            for file in os.listdir(output_folder):
                if base_name in file:
                    file_path = os.path.join(output_folder, file)
                    os.remove(file_path)

        print(f"Processing video: {video_file}...")
        t0 = time.time()
        write_video_ffmpeg(
            video_path,
            pause_image,
            output_folder,
            nvidia_gpu=nvidia_gpu,
            pc_client=pc_client,
        )
        # fallback option if ffmpeg fails
        # process_and_output_video(video_path, pause_image, output_folder)
        t1 = time.time()
        print(f"Finished processing {video_file}. Videos saved in {output_folder}.")
        print(f"Time taken for processing and output: {t1-t0:.2f} seconds.")


def process_and_output_video(video_path, pause_image, output_folder):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    paused_gray = cv2.cvtColor(pause_image, cv2.COLOR_BGR2GRAY)
    writer = cv2.VideoWriter(
        os.path.join(output_folder, "output.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),  # or another codec
        fps,
        (w, h),
    )
    frame = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break
        is_detected = compare_white_pixels(img, paused_gray)
        if not is_detected:
            writer.write(img)
        if frame % (int(fps) * 60) == 0:
            print(f"Processed {frame/(int(fps)*60)} minutes...")
        frame += 1


def write_video_ffmpeg(
    video_path, pause_image, output_folder, nvidia_gpu=False, pc_client=False
):
    output_params_nvidia = {
        "vcodec": "h264_nvenc",
        "pix_fmt": "yuv420p",
        "preset": "p6",
        "cq": 23,
    }
    output_params = {
        "vcodec": "libx264",
        "pix_fmt": "yuv420p",
        "preset": "medium",
        "crf": 18,
    }
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    paused_gray = cv2.cvtColor(pause_image, cv2.COLOR_BGR2GRAY)
    process = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="bgr24",
            s=f"{w}x{h}",
            framerate=fps,
        )
        .output(
            os.path.join(output_folder, "output.mp4"),
            **(output_params_nvidia if nvidia_gpu else output_params),
        )
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    frame = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break
        is_detected = compare_white_pixels(img, paused_gray, pc_client=pc_client)
        if not is_detected:
            process.stdin.write(img.tobytes())
        if frame % (int(fps) * 60) == 0:
            print(f"Processed {frame/(int(fps)*60)} minutes...")
        frame += 1
    process.stdin.close()
    process.wait()


def write_video_opencv(video_path, detection_results):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        "output.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),  # or another codec
        fps,
        (w, h),
    )

    frame = 0

    while True:
        ret, img = cap.read()
        if not ret:
            break
        if not detection_results[frame]:
            writer.write(img)
        if frame % (int(fps) * 60) == 0:
            print(f"Processed {frame/(int(fps)*60)} minutes...")
        frame += 1


# TODO: Use a gpu-neutral approach
if __name__ == "__main__":
    # Process the videos in the TODO folder and save the output in the output_videos folder
    # Change nvidia_gpu to True if you have a NVIDIA GPU for ffmpeg
    # Change pc_client to True if the recording is of the PC client
    split_and_rewrite_video(
        input_folder="TODO",
        output_parent_folder="output_videos",
        nvidia_gpu=False,
        pc_client=False,
    )
