'''
Frame level normalization at the flow vs frame level normalization at the magnitude
'''
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# breakpoint()
from load_flows import load_rgb_and_flow_from_directory
from flow_viz import flow_to_image
from datetime import datetime
import cv2

# start_frame = 137
# end_frame = 151
# cam = 'front'
# cat = '11'
# category_name = f'{cat}-same-{cam}-{start_frame}-{end_frame}'
# rgb_images, flow_data = load_rgb_and_flow_from_directory(category_name)
# breakpoint()
# frame_range = range(start_frame, end_frame + 1)

def get_max_flow_singular(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    # print("rad_max",rad_max) # stack wine 85.07742
    # print("rad_average", np.average(rad)) # stack wine 3.681587
    return rad_max

def get_seq_norm_flow(flow, max_flow):
    epsilon = 1e-5  # Small value to avoid division by zero
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    
    # Normalize the flow components
    u_normalized = u / (max_flow + epsilon)
    v_normalized = v / (max_flow + epsilon)
    
    # Create a new array to store the normalized flow
    normalized_flow = np.zeros_like(flow)
    normalized_flow[:, :, 0] = u_normalized
    normalized_flow[:, :, 1] = v_normalized
    
    return normalized_flow

def get_frame_norm_flow(flow):
    u = flow[:,:,0]
    v = flow[:,:,1]
    max_flow = get_max_flow_singular(flow)
    epsilon = 1e-5
    u = u / (max_flow + epsilon)
    v = v / (max_flow + epsilon)
    flow[:,:,0] = u
    flow[:,:,1] = v
    return flow

def get_flow_magnitudes(flow):
    u = flow[:,:,0]
    v = flow[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    return rad

    
def calculate_centroid(binary_mask):
    # Ensure the binary mask is binary
    binary_mask = binary_mask.astype(np.uint8)
    
    # Calculate moments
    moments = cv2.moments(binary_mask)
    
    # Calculate x, y coordinate of center
    if moments["m00"] != 0:
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
    else:
        # Set values as zero
        cX, cY = 0, 0

    return (cX, cY)


def connected_components(dilated_gt_seg, hand_seg, category_name, max_distance = 200):
    # Use connected components to get closest segments to gripper
    num_labels, labels, _, centroids = cv2.connectedComponentsWithStats(dilated_gt_seg)
    label_dir = os.path.join('tsg', f'labels', category_name) 
    os.makedirs(label_dir, exist_ok=True) 
    plt.imshow(labels)
    # plt.plot(hand_center_x,hand_center_y,'ro',markersize=16)
    breakpoint()
    hand_center_x, hand_center_y = calculate_centroid(hand_seg)
    # plt.plot(hand_center_x,hand_center_y,'ro',markersize=16)
    plt.savefig(os.path.join(label_dir, f'{category_name}.png'))

    filtered_labels = np.zeros_like(labels, dtype=np.uint8)
    for i in range(1, num_labels):
        c_x, c_y = centroids[i]
        # plt.plot(c_x, c_y, 'ro',markersize=16)
        distance = np.sqrt((c_x - hand_center_x)**2 + (c_y - hand_center_y)**2)
        print(distance)
        if distance <= max_distance:
            filtered_labels[labels == i] = 255
        # Check if the distance is within the specified maximum distanc
    # Create an empty binary mask with the same shape as the labeled image
    label_dir = os.path.join('tsg', f'post_conn_comp', category_name) 
    os.makedirs(label_dir, exist_ok=True) 
    # plt.plot(gripper_pos[0],gripper_pos[1],'ro',markersize=5)
    plt.imsave(os.path.join(label_dir, f'{category_name}.png'), filtered_labels)
    # Iterate through each selected segment and include its pixels in the combined mask
    return filtered_labels

# Exp 5 Sequence level norm followed by frame level norm, sequence level outlier removal
base_directory = '/home/jess/TaskSeg/tsg/jpgs'

# Use glob to find all directories that start with "12"
pattern = os.path.join(base_directory, 'JessConditioner2*')

# Iterate over each matching directory
val = 0
for dir_path in glob.glob(pattern):
    if os.path.isdir(dir_path):
        category_name = os.path.basename(dir_path)
        hand_seg = np.load('/home/jess/Downloads/H.npy')
        inv_hand_seg = np.invert(hand_seg)
        robot_seg = np.load('/home/jess/Downloads/person.npy')
        robot_seg = np.invert(robot_seg)
        breakpoint()
        rgb_images, flow_data = load_rgb_and_flow_from_directory(category_name)
        # breakpoint()
        parts = dir_path.split('-')
        print(f'parts: {parts}')
        start_frame = int(parts[2])
        end_frame = float(parts[3])
        frame_range = range(int(start_frame), int(end_frame+1))
        print(f'frame range: {frame_range}')
        mag_sum = np.zeros((rgb_images[0].shape[0], rgb_images[0].shape[1]))
        max_flow = -1000
        for idx in frame_range:
            value = idx - frame_range.start
            print(value)
            curr_max = get_max_flow_singular(flow_data[value])
            nn_save_dir = os.path.join('tsg', f'no_norm_flows', category_name) 
            os.makedirs(nn_save_dir, exist_ok=True) 
            plt.imsave(os.path.join(nn_save_dir, f'{category_name}-{value}.png'), flow_to_image(flow_data[idx - frame_range.start]) / 255.0)
            if curr_max > max_flow:
                max_flow = curr_max

        for idx in frame_range:
            value = idx - frame_range.start
            optical_flow = get_seq_norm_flow(flow_data[idx - frame_range.start], max_flow)
            fl_save_dir = os.path.join('tsg', f'seq_norm_flows', category_name) 
            os.makedirs(fl_save_dir, exist_ok=True) 
            plt.imsave(os.path.join(fl_save_dir, f'{category_name}-{value}.png'), flow_to_image(optical_flow) / 255.0)
            magnitudes = get_flow_magnitudes(optical_flow)
            mag_no_robot = np.multiply(magnitudes, robot_seg)
            mag_no_robot = np.multiply(mag_no_robot, inv_hand_seg)
            if mag_no_robot.max() > 0:
                curr_mag = mag_no_robot/np.max(mag_no_robot)
                framemag_save_dir = os.path.join('tsg', f'seq_and_frame_norm_mag', category_name) 
                os.makedirs(framemag_save_dir, exist_ok=True) 
                plt.imsave(os.path.join(framemag_save_dir, f'{category_name}-{value}.png'), curr_mag)
                mag_sum += curr_mag
            # mag_no_robot = np.multiply(magnitudes, robot_seg)
            # curr_mag = mag_no_robot/ #np.max(mag_no_robot)

        # save the mags
        mag_save_dir = os.path.join('tsg', f'magnitudes_seq_norm_at_flow_frame_at_mag_seq_outlier_rem', category_name) 
        os.makedirs(mag_save_dir, exist_ok=True) 
        plt.imsave(os.path.join(mag_save_dir, f'{category_name}.png'), mag_sum)
        np.save(os.path.join(mag_save_dir, f'{category_name}.npy'), mag_sum)

        # breakpoint()
        # Compute the average magnitude over the frame range
        norm_mag_sum = mag_sum / len(frame_range)

        data = norm_mag_sum.flatten()
        mean = np.mean(data)
        median = np.median(data)
        std = np.std(data)

        kernel = np.ones((5, 5), np.uint8)
        ground_truth_seg_loose = (norm_mag_sum >= mean + 1.4 * std).astype(np.uint8)
        breakpoint()
        eroded_gt_seg_loose = cv2.erode(ground_truth_seg_loose, kernel, iterations=1)
        dilated_gt_seg_loose = cv2.dilate(eroded_gt_seg_loose, kernel, iterations=1)

        seg_save_dir = os.path.join('tsg', 'final_seg-5', category_name)
        os.makedirs(seg_save_dir, exist_ok=True)
        plt.imsave(os.path.join(seg_save_dir, f'{0:05}.png'), dilated_gt_seg_loose)
        plt.clf()
        plt.figure(figsize=(16.63, 16.63))
        count = (dilated_gt_seg_loose == 1).sum()
        x_center, y_center = calculate_centroid(dilated_gt_seg_loose)#np.argwhere(dilated_gt_seg_loose==1).sum(0)/count

        final_gt_seg_l = connected_components(dilated_gt_seg_loose, hand_seg, category_name)

        non_zero_coords = np.nonzero(final_gt_seg_l)

        # Bounding box coordinates
        y_min, x_min = np.min(non_zero_coords[0]), np.min(non_zero_coords[1])
        y_max, x_max = np.max(non_zero_coords[0]), np.max(non_zero_coords[1])
        # Width and height of the bounding box
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        plt.imshow(rgb_images[0])
        plt.imshow(final_gt_seg_l, alpha=0.5)
        # breakpoint()
        plt.grid(False)
        plt.axis('off')
        # plt.plot([x_min, x_min], [y_min, y_max], color='red')
        # plt.plot([x_max, x_max], [y_min, y_max], color='red')
        # plt.plot([x_min, x_max], [y_min, y_min], color='red')
        # plt.plot([x_min, x_max], [y_max, y_max], color='red')
        # plt.plot(x_center, y_center, 'r*', markersize=16)
        print(f'x: {x_center}, y: {y_center}')
        os.makedirs(f'/home/jess/TaskSeg/final_masks-5/segs/jess', exist_ok=True) 
        os.makedirs(f'/home/jess/TaskSeg/final_masks-5/images/jess', exist_ok=True) 
        os.makedirs(f'/home/jess/TaskSeg/final_masks-5/gt_bboxs/jess', exist_ok=True) 
        os.makedirs(f'/home/jess/TaskSeg/final_masks-5/jess', exist_ok=True) 
        np.save(f'/home/jess/TaskSeg/final_masks-5/gt_bboxs/jess/{val}.npy', np.array([x_min, y_min, x_max, y_max]))
        np.save(f'/home/jess/TaskSeg/final_masks-5/segs/jess/{val}.npy', final_gt_seg_l)
        plt.savefig(f'/home/jess/TaskSeg/final_masks-5/jess/{val}.png')
        plt.clf()
        plt.imsave(f'/home/jess/TaskSeg/final_masks-5/images/jess/{val}.png', rgb_images[0])
        val+=1