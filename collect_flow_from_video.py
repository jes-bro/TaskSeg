import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import cv2
import ffmpeg
import matplotlib.pyplot as plt
import copy
from PIL import Image
import torch
import sys
sys.path.append('core')
import logging
import glob
import os
import argparse
from datetime import datetime
from raft import RAFT
from flow_viz import flow_to_image, get_max_flow_singular, flow_to_image_sequential
from utils import flow_viz
from utils.utils import InputPadder
from typing import List

import torchvision as tv
from torchvision import transforms as T
import matplotlib.pyplot as plt

import importlib

DEVICE = "cuda:0"

def convert_vid_to_rgbs(vid_path: str):
    # Use ffmpeg to decode the video
    process = (
        ffmpeg
        .input(vid_path)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )

    # Get video metadata (e.g., width, height)
    probe = ffmpeg.probe(vid_path)
    video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])

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

def load_image(imfile):
    img = np.array(imfile).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def get_pred_flow(obs_1, obs_2, raft_model):

    with torch.no_grad():
        image1 = load_image(obs_1)
        image2 = load_image(obs_2)

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_up = raft_model(image1, image2, iters=40, test_mode=True)
        return flow_up

def keypoint_discovery(demo) -> List[int]:
    demo_len = len(demo)
    keyframes = []
    for i in range(0, demo_len, 200):
        keyframes.append(i)
    return keyframes

def generate_pseudo_gt(demo, keypoint_1, keypoint_2, raft_model):

    frame_range = range(keypoint_1+1, keypoint_2+1) # Start prediction from subsequent frame to next keyframe

    # Save robot segmentation
    fstop = frame_range.stop - 1
    category_name = f"JessConditioner2-same-{frame_range.start}-{fstop}-{str(datetime.now())}"
    for idx in frame_range:
        frame_n = demo[idx]
        # breakpoint()
        frame_n = frame_n.reshape(demo[0].shape[0],demo[0].shape[1], 3)
        # Use RAFT to get optical flow
        optical_flow = get_pred_flow(demo[0], frame_n, raft_model)
        optical_flow = optical_flow.squeeze().permute(1, 2, 0)

        # Save RGB image 
        value = idx - frame_range.start
        rgb_save_dir = os.path.join('tsg', 'jpgs', category_name) 
        os.makedirs(rgb_save_dir, exist_ok=True) 
        plt.imsave(os.path.join(rgb_save_dir, f'{value:05}.jpg'), demo[idx]) 
        plt.clf()

        # Save Flow
        flow_save_dir = os.path.join('tsg', f'non_normalized_flow', category_name) 
        os.makedirs(flow_save_dir, exist_ok=True) 
        print("Saving flow seq")
        np.save(os.path.join(flow_save_dir, f'{value:05}.npy'), optical_flow.cpu().numpy())
        flow_save_dir = os.path.join('tsg', f'non_normalized_flow_imgs', category_name) 
        os.makedirs(flow_save_dir, exist_ok=True) 
        plt.imsave(os.path.join(flow_save_dir, f'{value:05}.jpg'), flow_to_image(optical_flow.cpu().numpy())) 
        plt.clf()

def get_keypoint_pseudo_gt(demo, keypoint_1, keypoint_2, raft_model):
    generate_pseudo_gt(demo, keypoint_1, keypoint_2, raft_model)
    
def get_flow_and_jpgs_from_video(demo_path: str):

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/raft-things.pth', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--threshold', default=0.001, help='threshold') # Use 0.05 if want a tighter threhold for flow aggregation (currently collecting both loose and tight pseudo ground truth)
    args = parser.parse_args()

    raft_model = torch.nn.DataParallel(RAFT(args))
    raft_model.load_state_dict(torch.load(args.model))
    raft_model = raft_model.module
    raft_model.to(DEVICE)
    raft_model.eval()

    demo = convert_vid_to_rgbs(demo_path)
    # breakpoint()
    # Get keypoints
    keypoints = keypoint_discovery(demo)

    # For each keypoint, get pseudo ground truth
    for keypoint_idx, keypoint in enumerate(keypoints):
        if keypoint_idx == len(keypoints)-1:
            get_keypoint_pseudo_gt(demo, keypoint, len(demo)-1, raft_model)
        else:
            get_keypoint_pseudo_gt(demo, keypoint, keypoints[keypoint_idx+1], raft_model)
        
get_flow_and_jpgs_from_video(demo_path="/home/jess/Downloads/jess_conditioner.mp4")
    