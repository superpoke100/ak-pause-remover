import argparse

from pause_remover.process_videos import split_and_rewrite_video

# TODO: Use a gpu-neutral approach
if __name__ == "__main__":
    # Process the videos in the TODO folder and save the output in the output_videos folder
    parser = argparse.ArgumentParser(
        prog="AK Pause Remover",
        description="Remove pauses from AK stage recordings",
    )
    parser.add_argument(
        "--emu",
        action="store_true",
        help="Configure detection for MuMu Emulator. Default is PC.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use NVIDIA GPU video encoder. Default is CPU.",
    )
    args = parser.parse_args()
    split_and_rewrite_video(
        input_foldername="TODO",
        output_foldername="output_videos",
        nvidia_gpu=args.gpu,
        pc_client=not args.emu,
    )
