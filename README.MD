## USAGE
1. Place videos to process in the TODO folder.
2. Run process_videos.py with desired options (see below)
3. Processed videos will be placed in the folder output_videos/{VIDEO_NAME}/output.mp4 once finished.

## Supported Configurations
- **MuMu Emulator**: 1920x1080 resolution, native recording feature
- **Official PC Client**: 1920x1080
- If using other recording software, ensure that UI elements of the emulator are not captured.

## OPTIONS
```
usage: AK Pause Remover [-h] [--emu] [--gpu]

options:
  -h, --help  show this help message and exit
  --emu       Configure detection for MuMu Emulator. Default is PC.
  --gpu       Use NVIDIA GPU video encoder. Default is CPU.
```

## Credits
Tums for the original version of this code.


