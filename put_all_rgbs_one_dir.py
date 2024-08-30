import numpy as np
import ffmpeg
import os
import glob
import matplotlib.pyplot as plt
from typing import List 

def convert_vid_to_rgbs(vid_path: str) -> List[np.ndarray]:
    # Get video metadata (e.g., width, height)
    probe = ffmpeg.probe(vid_path)
    video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])

    # Use ffmpeg to decode the video
    process = (
        ffmpeg
        .input(vid_path)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )

    # Read raw video frames
    frame_size = width * height * 3
    frames = []
    while True:
        in_bytes = process.stdout.read(frame_size)
        if not in_bytes:
            break
        frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )
        frames.append(frame)

    return frames

def save_frames(frames: List[np.ndarray], output_dir: str, start_index: int) -> int:
    os.makedirs(output_dir, exist_ok=True)
    for idx, frame in enumerate(frames):
        frame_save_path = os.path.join(output_dir, f"{start_index + idx}.jpg")
        plt.imsave(frame_save_path, frame)
        plt.clf()
    return start_index + len(frames)

def process_videos_in_directory(demo_path: str, output_path: str):
    global_frame_index = 0
    for vid_path in glob.glob(demo_path):
        frames = convert_vid_to_rgbs(vid_path)
        global_frame_index = save_frames(frames, output_path, global_frame_index)
        print(f"Processed and saved frames from video: {os.path.basename(vid_path)}")

if __name__ == "__main__":
    demo_path = "/home/jess/Downloads/ziyu.mp4"  # Update with your path
    output_path = "/home/jess/ziyu"  # Update with your output directory
    process_videos_in_directory(demo_path, output_path)
