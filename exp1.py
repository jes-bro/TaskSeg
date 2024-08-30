'''
Frame level normalization at the flow vs frame level normalization at the magnitude
'''
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# breakpoint()
from skimage.transform import resize
from load_flows import load_rgb_and_flow_from_directory_no_resize, load_rgb_and_flow_from_directory
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
    # print("rad_max",rad_max) # stack LIGHT 85.07742
    # print("rad_average", np.average(rad)) # stack LIGHT 3.681587
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


def connected_components(dilated_gt_seg, gripper_pos, category_name, max_distance = 600):
    # Use connected components to get closest segments to gripper
    num_labels, labels, _, centroids = cv2.connectedComponentsWithStats(dilated_gt_seg)
    label_dir = os.path.join('tsg', f'labels', category_name) 
    os.makedirs(label_dir, exist_ok=True) 
    plt.imshow(labels)
    # plt.plot(gripper_pos[0],gripper_pos[1],'ro',markersize=5)
    plt.savefig(os.path.join(label_dir, f'{category_name}.png'))

    filtered_labels = np.zeros_like(labels, dtype=np.uint8)
    for i in range(1, num_labels):
        c_x, c_y = centroids[i]
        distance = np.sqrt((c_x - gripper_pos[0])**2 + (c_y - gripper_pos[1])**2)
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

############################################################################################################################################
# Exp 0: frame wise normalization at the flow level, sequence level outlier removal

# mag_sum = np.zeros((rgb_images[0].shape[0], rgb_images[0].shape[1]))
# # # breakpoint()
# robot_seg = np.load(f'/home/jess/TaskSeg/tsg/robot_seg/{cat}-same-{cam}-{start_frame}-{end_frame}/00000.npy')
# ## Frame wise norm at flow
# for idx in frame_range:
#     optical_flow = get_frame_norm_flow(flow_data[idx - frame_range.start])
#     magnitudes = get_flow_magnitudes(optical_flow)
#     # breakpoint()
#     mag_no_robot = np.multiply(magnitudes, robot_seg)
#     if mag_no_robot.max() > 0:
#         mag_sum += mag_no_robot #/np.max(mag_no_robot)

# # save the mags
# mag_save_dir = os.path.join('tsg', f'magnitudes_frame_norm_at_flow_seq_outlier_rem', category_name) 
# os.makedirs(mag_save_dir, exist_ok=True) 
# # breakpoint()
# plt.imsave(os.path.join(mag_save_dir, f'{category_name}.png'), mag_sum)
# np.save(os.path.join(mag_save_dir, f'{category_name}.npy'), mag_sum)

# # Compute the average magnitude over the frame range
# norm_mag_sum = mag_sum / len(frame_range)

# data = norm_mag_sum.flatten()
# mean = np.mean(data)
# median =np.median(data)
# std = np.std(data)

# # Filter values above mean + n * std (where n ranges from 1 to 10)
# for n in range(1, 11):
#     threshold = mean + n * std
#     filtered_mag_sum = np.copy(norm_mag_sum)
#     filtered_mag_sum[filtered_mag_sum < threshold] = np.nan

#     # Plot the filtered magnitude sum image
#     plt.figure(figsize=(10, 6))
#     plt.imshow(filtered_mag_sum, cmap='viridis')
#     plt.colorbar()
#     plt.title(f'Filtered Magnitude Sum (values >= mean + {n} * std)')
#     plt.axis('off')
#     plt.savefig(os.path.join(mag_save_dir, f'{category_name}-{n}.png'))

# cmap = plt.get_cmap('viridis')

# # Normalize the data for color mapping
# norm = mcolors.Normalize(vmin=np.min(data), vmax=np.max(data))

# # Create the histogram data
# n, bins = np.histogram(data, bins=100)

# # Scatter plot for the histogram
# fig, ax = plt.subplots(figsize=(10, 6))

# # Create a scatter plot of the data points with log scale
# # Calculate the bin centers
# bin_centers = 0.5 * (bins[:-1] + bins[1:])
# bin_counts, _ = np.histogram(data, bins=bins)

# # Create a scatter plot of the histogram
# sc = ax.scatter(bin_centers, bin_counts, c=bin_centers, cmap=cmap, norm=norm, edgecolor='black', alpha=0.6)

# # Set y-axis to log scale
# ax.set_yscale('log')

# # Add colorbar to the scatter plot
# cbar = plt.colorbar(sc, ax=ax)
# cbar.set_label('Magnitude')

# # Add mean and median lines
# plt.axvline(mean, color='r', linestyle='dashed', linewidth=1, label='Mean')
# plt.axvline(median, color='g', linestyle='dashed', linewidth=1, label='Median')


# # Bootstrapping for the standard error of the median
# n_bootstrap_samples = 1000
# bootstrapped_medians = np.empty(n_bootstrap_samples)

# for i in range(n_bootstrap_samples):
#     bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
#     bootstrapped_medians[i] = np.median(bootstrap_sample)

# se_median = np.std(bootstrapped_medians)

# # Add standard deviation lines from +1 STD to +5 STD
# for i in range(1, 11):
#     plt.axvline(mean + i * std, color='r', linestyle='dotted', linewidth=1, label=f'Mean + {i} STD' if i == 1 else "")
#     plt.axvline(median + i * se_median, color='g', linestyle='dotted', linewidth=1, label=f'Median + {i} SE' if i == 1 else "")

# plt.legend()
# plt.title('Histogram of Magnitudes')
# plt.xlabel('Magnitude')
# plt.ylabel('Frequency (Log Scale)')
# plt.grid(True, which='both', linestyle='--')

# # Save the histogram
# hist_save_dir = os.path.join('tsg', 'hist', category_name)
# os.makedirs(hist_save_dir, exist_ok=True)
# plt.savefig(os.path.join(hist_save_dir, f'{0:05}.png'))

# kernel = np.ones((5, 5), np.uint8)
# ground_truth_seg_loose = (norm_mag_sum >= mean + 8 * std).astype(np.uint8)
# eroded_gt_seg_loose = cv2.erode(ground_truth_seg_loose, kernel, iterations=1)
# dilated_gt_seg_loose = cv2.dilate(eroded_gt_seg_loose, kernel, iterations=1)

# seg_save_dir = os.path.join('tsg', 'final_seg', category_name)
# os.makedirs(seg_save_dir, exist_ok=True)
# plt.imsave(os.path.join(seg_save_dir, f'{0:05}.png'), dilated_gt_seg_loose)
# plt.clf()

# plt.imshow(rgb_images[0])
# plt.imshow(dilated_gt_seg_loose, alpha=0.5)
# os.makedirs(f'final_masks/{category_name}/{0:05}', exist_ok=True) 
# plt.savefig(f'final_masks/{category_name}/{0:05}/'+str(datetime.now())+'.png',bbox_inches='tight', pad_inches=0)
# plt.clf()

# ##############################################################################################################################################
# ## EXP 2 framewise norm at magnitude agg, sequence level outlier removal
# mag_sum = np.zeros((rgb_images[0].shape[0], rgb_images[0].shape[1]))

# # Frame wise norm at magnitude agg
# for idx in frame_range:
#     optical_flow = flow_data[idx - frame_range.start]
#     magnitudes = get_flow_magnitudes(optical_flow)
#     mag_no_robot = np.multiply(magnitudes, robot_seg)
#     if mag_no_robot.max() > 0:
#         mag_sum += mag_no_robot / np.max(mag_no_robot)

# # save the mags
# mag_save_dir = os.path.join('tsg', f'magnitudes_frame_norm_at_mag_seq_outlier_rem', category_name) 
# os.makedirs(mag_save_dir, exist_ok=True) 
# plt.imsave(os.path.join(mag_save_dir, f'{category_name}.png'), mag_sum)
# np.save(os.path.join(mag_save_dir, f'{category_name}.npy'), mag_sum)


# # Compute the average magnitude over the frame range
# norm_mag_sum = mag_sum / len(frame_range)

# data = norm_mag_sum.flatten()
# mean = np.mean(data)
# median =np.median(data)
# std = np.std(data)

# # Filter values above mean + n * std (where n ranges from 1 to 10)
# for n in range(1, 11):
#     threshold = mean + n * std
#     filtered_mag_sum = np.copy(norm_mag_sum)
#     filtered_mag_sum[filtered_mag_sum < threshold] = np.nan

#     # Plot the filtered magnitude sum image
#     plt.figure(figsize=(10, 6))
#     plt.imshow(filtered_mag_sum, cmap='viridis')
#     plt.colorbar()
#     plt.title(f'Filtered Magnitude Sum (values >= mean + {n} * std)')
#     plt.axis('off')
#     plt.savefig(os.path.join(mag_save_dir, f'{category_name}-{n}.png'))

# cmap = plt.get_cmap('viridis')

# # Normalize the data for color mapping
# norm = mcolors.Normalize(vmin=np.min(data), vmax=np.max(data))

# # Create the histogram data
# n, bins = np.histogram(data, bins=100)

# # Scatter plot for the histogram
# fig, ax = plt.subplots(figsize=(10, 6))

# # Create a scatter plot of the data points with log scale
# # Calculate the bin centers
# bin_centers = 0.5 * (bins[:-1] + bins[1:])
# bin_counts, _ = np.histogram(data, bins=bins)

# # Create a scatter plot of the histogram
# sc = ax.scatter(bin_centers, bin_counts, c=bin_centers, cmap=cmap, norm=norm, edgecolor='black', alpha=0.6)

# # Set y-axis to log scale
# ax.set_yscale('log')

# # Add colorbar to the scatter plot
# cbar = plt.colorbar(sc, ax=ax)
# cbar.set_label('Magnitude')

# # Add mean and median lines
# plt.axvline(mean, color='r', linestyle='dashed', linewidth=1, label='Mean')
# plt.axvline(median, color='g', linestyle='dashed', linewidth=1, label='Median')

# # Add standard deviation lines from +1 STD to +5 STD
# for i in range(1, 11):
#     plt.axvline(mean + i * std, color='r', linestyle='dotted', linewidth=1, label=f'Mean + {i} STD' if i == 1 else "")
#     plt.axvline(median + i * std, color='g', linestyle='dotted', linewidth=1, label=f'Median + {i} STD' if i == 1 else "")

# plt.legend()
# plt.title('Histogram of Magnitudes')
# plt.xlabel('Magnitude')
# plt.ylabel('Frequency')
# plt.grid(True, which='both', linestyle='--')

# # Save the histogram
# hist_save_dir = os.path.join('tsg', 'hist-2', category_name)
# os.makedirs(hist_save_dir, exist_ok=True)
# plt.savefig(os.path.join(hist_save_dir, f'{0:05}.png'))

# kernel = np.ones((5, 5), np.uint8)
# ground_truth_seg_loose = (norm_mag_sum >= mean + 8 * std).astype(np.uint8)
# eroded_gt_seg_loose = cv2.erode(ground_truth_seg_loose, kernel, iterations=1)
# dilated_gt_seg_loose = cv2.dilate(eroded_gt_seg_loose, kernel, iterations=1)

# seg_save_dir = os.path.join('tsg', 'final_seg-2', category_name)
# os.makedirs(seg_save_dir, exist_ok=True)
# plt.imsave(os.path.join(seg_save_dir, f'{0:05}.png'), dilated_gt_seg_loose)
# plt.clf()

# plt.imshow(rgb_images[0])
# plt.imshow(dilated_gt_seg_loose, alpha=0.5)
# os.makedirs(f'final_masks-2/{category_name}/{0:05}', exist_ok=True) 
# plt.savefig(f'final_masks-2/{category_name}/{0:05}/'+str(datetime.now())+'.png',bbox_inches='tight', pad_inches=0)
# plt.clf()



# ###############################################################################################################################################
# # Exp 3 sequence level norm at flow level, sequence level norm

# mag_sum = np.zeros((rgb_images[0].shape[0], rgb_images[0].shape[1]))

# # Sequence level norm at flow
# max_flow = -1000
# for idx in frame_range:
#     curr_max = get_max_flow_singular(flow_data[idx - frame_range.start])
#     if curr_max > max_flow:
#         max_flow = curr_max

# for idx in frame_range:
#     optical_flow = get_seq_norm_flow(flow_data[idx - frame_range.start], max_flow)
#     magnitudes = get_flow_magnitudes(optical_flow)
#     # breakpoint()
#     mag_no_robot = np.multiply(magnitudes, robot_seg)
#     if mag_no_robot.max() > 0:
#         mag_sum += mag_no_robot #/np.max(mag_no_robot)

# # save the mags
# mag_save_dir = os.path.join('tsg', f'magnitudes_seq_norm_at_flow_seq_outlier_rem', category_name) 
# os.makedirs(mag_save_dir, exist_ok=True) 
# plt.imsave(os.path.join(mag_save_dir, f'{category_name}.png'), mag_sum)
# np.save(os.path.join(mag_save_dir, f'{category_name}.npy'), mag_sum)


# # Compute the average magnitude over the frame range
# norm_mag_sum = mag_sum / len(frame_range)

# data = norm_mag_sum.flatten()
# mean = np.mean(data)
# median =np.median(data)
# std = np.std(data)

# # Filter values above mean + n * std (where n ranges from 1 to 10)
# for n in range(1, 11):
#     threshold = mean + n * std
#     filtered_mag_sum = np.copy(norm_mag_sum)
#     filtered_mag_sum[filtered_mag_sum < threshold] = np.nan

#     # Plot the filtered magnitude sum image
#     plt.figure(figsize=(10, 6))
#     plt.imshow(filtered_mag_sum, cmap='viridis')
#     plt.colorbar()
#     plt.title(f'Filtered Magnitude Sum (values >= mean + {n} * std)')
#     plt.axis('off')
#     plt.savefig(os.path.join(mag_save_dir, f'{category_name}-{n}.png'))

# cmap = plt.get_cmap('viridis')

# # Normalize the data for color mapping
# norm = mcolors.Normalize(vmin=np.min(data), vmax=np.max(data))

# # Create the histogram data
# n, bins = np.histogram(data, bins=100)

# # Scatter plot for the histogram
# fig, ax = plt.subplots(figsize=(10, 6))

# # Create a scatter plot of the data points with log scale
# # Calculate the bin centers
# bin_centers = 0.5 * (bins[:-1] + bins[1:])
# bin_counts, _ = np.histogram(data, bins=bins)

# # Create a scatter plot of the histogram
# sc = ax.scatter(bin_centers, bin_counts, c=bin_centers, cmap=cmap, norm=norm, edgecolor='black', alpha=0.6)

# # Set y-axis to log scale
# ax.set_yscale('log')

# # Add colorbar to the scatter plot
# cbar = plt.colorbar(sc, ax=ax)
# cbar.set_label('Magnitude')

# # Add mean and median lines
# plt.axvline(mean, color='r', linestyle='dashed', linewidth=1, label='Mean')
# plt.axvline(median, color='g', linestyle='dashed', linewidth=1, label='Median')

# # Add standard deviation lines from +1 STD to +5 STD
# for i in range(1, 11):
#     plt.axvline(mean + i * std, color='r', linestyle='dotted', linewidth=1, label=f'Mean + {i} STD' if i == 1 else "")
#     plt.axvline(median + i * std, color='g', linestyle='dotted', linewidth=1, label=f'Median + {i} STD' if i == 1 else "")

# plt.legend()
# plt.title('Histogram of Magnitudes')
# plt.xlabel('Magnitude')
# plt.ylabel('Frequency')
# plt.grid(True, which='both', linestyle='--')

# # Save the histogram
# hist_save_dir = os.path.join('tsg', 'hist-3', category_name)
# os.makedirs(hist_save_dir, exist_ok=True)
# plt.savefig(os.path.join(hist_save_dir, f'{0:05}.png'))

# kernel = np.ones((5, 5), np.uint8)
# ground_truth_seg_loose = (norm_mag_sum >= mean + 8 * std).astype(np.uint8)
# eroded_gt_seg_loose = cv2.erode(ground_truth_seg_loose, kernel, iterations=1)
# dilated_gt_seg_loose = cv2.dilate(eroded_gt_seg_loose, kernel, iterations=1)

# seg_save_dir = os.path.join('tsg', 'final_seg-3', category_name)
# os.makedirs(seg_save_dir, exist_ok=True)
# plt.imsave(os.path.join(seg_save_dir, f'{0:05}.png'), dilated_gt_seg_loose)
# plt.clf()

# plt.imshow(rgb_images[0])
# plt.imshow(dilated_gt_seg_loose, alpha=0.5)
# os.makedirs(f'final_masks-3/{category_name}/{0:05}', exist_ok=True) 
# plt.savefig(f'final_masks-3/{category_name}/{0:05}/'+str(datetime.now())+'.png',bbox_inches='tight', pad_inches=0)
# plt.clf()

# ############################################################################################################################################
# # Exp 4 sequence level norm at mag level, sequence level outlier removal

# mag_sum = np.zeros((rgb_images[0].shape[0], rgb_images[0].shape[1]))

# # Sequence level norm at magnitude
# max_flow = -1000
# for idx in frame_range:
#     curr_max = get_max_flow_singular(flow_data[idx - frame_range.start])
#     if curr_max > max_flow:
#         max_flow = curr_max

# for idx in frame_range:
#     optical_flow = flow_data[idx - frame_range.start]
#     magnitudes = get_flow_magnitudes(optical_flow)
#     # breakpoint()
#     mag_no_robot = np.multiply(magnitudes, robot_seg)
#     if mag_no_robot.max() > 0:
#         mag_sum += mag_no_robot / max_flow

# # save the mags
# mag_save_dir = os.path.join('tsg', f'magnitudes_seq_norm_at_mag_seq_outlier_rem', category_name) 
# os.makedirs(mag_save_dir, exist_ok=True) 
# plt.imsave(os.path.join(mag_save_dir, f'{category_name}.png'), mag_sum)
# np.save(os.path.join(mag_save_dir, f'{category_name}.npy'), mag_sum)


# # Compute the average magnitude over the frame range
# norm_mag_sum = mag_sum / len(frame_range)

# data = norm_mag_sum.flatten()
# mean = np.mean(data)
# median =np.median(data)
# std = np.std(data)

# # Filter values above mean + n * std (where n ranges from 1 to 10)
# for n in range(1, 11):
#     threshold = mean + n * std
#     filtered_mag_sum = np.copy(norm_mag_sum)
#     filtered_mag_sum[filtered_mag_sum < threshold] = np.nan

#     # Plot the filtered magnitude sum image
#     plt.figure(figsize=(10, 6))
#     plt.imshow(filtered_mag_sum, cmap='viridis')
#     plt.colorbar()
#     plt.title(f'Filtered Magnitude Sum (values >= mean + {n} * std)')
#     plt.axis('off')
#     plt.savefig(os.path.join(mag_save_dir, f'{category_name}-{n}.png'))

# cmap = plt.get_cmap('viridis')

# # Normalize the data for color mapping
# norm = mcolors.Normalize(vmin=np.min(data), vmax=np.max(data))

# # Create the histogram data
# n, bins = np.histogram(data, bins=100)

# # Scatter plot for the histogram
# fig, ax = plt.subplots(figsize=(10, 6))

# # Create a scatter plot of the data points with log scale
# # Calculate the bin centers
# bin_centers = 0.5 * (bins[:-1] + bins[1:])
# bin_counts, _ = np.histogram(data, bins=bins)

# # Create a scatter plot of the histogram
# sc = ax.scatter(bin_centers, bin_counts, c=bin_centers, cmap=cmap, norm=norm, edgecolor='black', alpha=0.6)

# # Set y-axis to log scale
# ax.set_yscale('log')

# # Add colorbar to the scatter plot
# cbar = plt.colorbar(sc, ax=ax)
# cbar.set_label('Magnitude')

# # Add mean and median lines
# plt.axvline(mean, color='r', linestyle='dashed', linewidth=1, label='Mean')
# plt.axvline(median, color='g', linestyle='dashed', linewidth=1, label='Median')

# # Add standard deviation lines from +1 STD to +5 STD
# for i in range(1, 11):
#     plt.axvline(mean + i * std, color='r', linestyle='dotted', linewidth=1, label=f'Mean + {i} STD' if i == 1 else "")
#     plt.axvline(median + i * std, color='g', linestyle='dotted', linewidth=1, label=f'Median + {i} STD' if i == 1 else "")

# plt.legend()
# plt.title('Histogram of Magnitudes')
# plt.xlabel('Magnitude')
# plt.ylabel('Frequency')
# plt.grid(True, which='both', linestyle='--')

# # Save the histogram
# hist_save_dir = os.path.join('tsg', 'hist-4', category_name)
# os.makedirs(hist_save_dir, exist_ok=True)
# plt.savefig(os.path.join(hist_save_dir, f'{0:05}.png'))

# kernel = np.ones((5, 5), np.uint8)
# ground_truth_seg_loose = (norm_mag_sum >= mean + 8 * std).astype(np.uint8)
# eroded_gt_seg_loose = cv2.erode(ground_truth_seg_loose, kernel, iterations=1)
# dilated_gt_seg_loose = cv2.dilate(eroded_gt_seg_loose, kernel, iterations=1)

# seg_save_dir = os.path.join('tsg', 'final_seg-4', category_name)
# os.makedirs(seg_save_dir, exist_ok=True)
# plt.imsave(os.path.join(seg_save_dir, f'{0:05}.png'), dilated_gt_seg_loose)
# plt.clf()

# plt.imshow(rgb_images[0])
# plt.imshow(dilated_gt_seg_loose, alpha=0.5)
# os.makedirs(f'final_masks-4/{category_name}/{0:05}', exist_ok=True) 
# plt.savefig(f'final_masks-4/{category_name}/{0:05}/'+str(datetime.now())+'.png',bbox_inches='tight', pad_inches=0)
# plt.clf()

####################################################################################################################################################
# Exp 5 Sequence level norm followed by frame level norm, sequence level outlier removal
base_directory = '/home/jess/TaskSeg/tsg/jpgs'

# Use glob to find all directories that start with "12"
pattern = os.path.join(base_directory, 'LIGHT*')

# Iterate over each matching directory
val = 0
for dir_path in glob.glob(pattern):
    if os.path.isdir(dir_path):
        category_name = os.path.basename(dir_path)
        # breakpoint()
        rgb_images, flow_data = load_rgb_and_flow_from_directory_no_resize(category_name)
        # breakpoint()
        parts = dir_path.split('-')
        start_frame = int(parts[3])
        end_frame = float(parts[4])
        frame_range = range(int(start_frame), int(end_frame + 1))

        mag_sum = np.zeros((rgb_images[0].shape[0], rgb_images[0].shape[1]))

        robot_seg = np.load(f'/home/jess/TaskSeg/tsg/robot_seg/{category_name}/00000.npy')
        # robot_seg = resize(robot_seg,(400, 400), preserve_range=True, anti_aliasing=True)
        gripper_pos = np.load(f'/home/jess/TaskSeg/tsg/gripper/{category_name}/00000.npy')
        # original_size = 1280
        # new_size = 400
        # scaling_factor = new_size / original_size

        # Scale the gripper positions
        # gripper_pos = gripper_pos * scaling_factor
        pcd_1 = np.load(f'/home/jess/TaskSeg/tsg/pcl/{category_name}/00000.npy')
        # original_size = 1280
        # new_size = 400
        # scaling_factor = new_size / original_size

        # Scale the point cloud positions
        # pcd_1 = pcd_1 * scaling_factor
        # Sequence level norm at flow, frame level norm 
        max_flow = -1000
        for idx in frame_range:
            value = idx - frame_range.start
            curr_max = get_max_flow_singular(flow_data[idx - frame_range.start])
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
            # breakpoint()
            mag_no_robot = np.multiply(magnitudes, robot_seg)
            if mag_no_robot.max() > 0:
                curr_mag = mag_no_robot/np.max(mag_no_robot)
                framemag_save_dir = os.path.join('tsg', f'seq_and_frame_norm_mag', category_name) 
                os.makedirs(framemag_save_dir, exist_ok=True) 
                plt.imsave(os.path.join(framemag_save_dir, f'{category_name}-{value}.png'), curr_mag)
                mag_sum += curr_mag

        # save the mags
        mag_save_dir = os.path.join('tsg', f'magnitudes_seq_norm_at_flow_frame_at_mag_seq_outlier_rem', category_name) 
        os.makedirs(mag_save_dir, exist_ok=True) 
        plt.imsave(os.path.join(mag_save_dir, f'{category_name}.png'), mag_sum)
        np.save(os.path.join(mag_save_dir, f'{category_name}.npy'), mag_sum)


        # Compute the average magnitude over the frame range
        norm_mag_sum = mag_sum / len(frame_range)

        data = norm_mag_sum.flatten()
        mean = np.mean(data)
        median = np.median(data)
        std = np.std(data)

        # # Filter values above mean + n * std (where n ranges from 1 to 10)
        
        # for n in range(1, 11):
        #     threshold = mean + n * std
        #     filtered_mag_sum = np.copy(norm_mag_sum)
        #     filtered_mag_sum[filtered_mag_sum < threshold] = np.nan

        #     # Plot the filtered magnitude sum image
        #     plt.figure(figsize=(10, 6))
        #     plt.imshow(filtered_mag_sum, cmap='viridis')
        #     plt.colorbar()
        #     plt.title(f'Filtered Magnitude Sum (values >= mean + {n} * std)')
        #     plt.axis('off')
        #     plt.savefig(os.path.join(mag_save_dir, f'{category_name}-{n}.png'))

        # cmap = plt.get_cmap('viridis')

        # # Normalize the data for color mapping
        # norm = mcolors.Normalize(vmin=np.min(data), vmax=np.max(data))

        # # Create the histogram data
        # n, bins = np.histogram(data, bins=100)

        # # Scatter plot for the histogram
        # fig, ax = plt.subplots(figsize=(10, 6))

        # # Create a scatter plot of the data points with log scale
        # # Calculate the bin centers
        # bin_centers = 0.5 * (bins[:-1] + bins[1:])
        # bin_counts, _ = np.histogram(data, bins=bins)

        # # Create a scatter plot of the histogram
        # sc = ax.scatter(bin_centers, bin_counts, c=bin_centers, cmap=cmap, norm=norm, edgecolor='black', alpha=0.6)

        # # Set y-axis to log scale
        # ax.set_yscale('log')

        # # Add colorbar to the scatter plot
        # cbar = plt.colorbar(sc, ax=ax)
        # cbar.set_label('Magnitude')

        # # Add mean and median lines
        # plt.axvline(mean, color='r', linestyle='dashed', linewidth=1, label='Mean')
        # plt.axvline(median, color='g', linestyle='dashed', linewidth=1, label='Median')

        # # Add standard deviation lines from +1 STD to +5 STD
        # for i in range(1, 11):
        #     plt.axvline(mean + i * std, color='r', linestyle='dotted', linewidth=1, label=f'Mean + {i} STD' if i == 1 else "")
        #     plt.axvline(median + i * std, color='g', linestyle='dotted', linewidth=1, label=f'Median + {i} STD' if i == 1 else "")

        # plt.legend()
        # plt.title('Histogram of Magnitudes')
        # plt.xlabel('Magnitude')
        # plt.ylabel('Frequency (Log scale)')
        # plt.grid(True, which='both', linestyle='--')

        # # Save the histogram
        # hist_save_dir = os.path.join('tsg', 'hist-5', category_name)
        # os.makedirs(hist_save_dir, exist_ok=True)
        # plt.savefig(os.path.join(hist_save_dir, f'{0:05}.png'))

        kernel = np.ones((5, 5), np.uint8)
        ground_truth_seg_loose = (norm_mag_sum >= mean + 4 * std).astype(np.uint8)
        eroded_gt_seg_loose = cv2.erode(ground_truth_seg_loose, kernel, iterations=1)
        dilated_gt_seg_loose = cv2.dilate(eroded_gt_seg_loose, kernel, iterations=1)

        seg_save_dir = os.path.join('tsg', 'final_seg-5', category_name)
        os.makedirs(seg_save_dir, exist_ok=True)
        plt.imsave(os.path.join(seg_save_dir, f'{0:05}.png'), dilated_gt_seg_loose)
        plt.clf()
        plt.figure(figsize=(16.63, 16.63))
        count = (dilated_gt_seg_loose == 1).sum()
        x_center, y_center = calculate_centroid(dilated_gt_seg_loose)#np.argwhere(dilated_gt_seg_loose==1).sum(0)/count
        for i in range(ground_truth_seg_loose.shape[0]):
            for j in range(ground_truth_seg_loose.shape[1]):
                if pcd_1[i][j][0] < -0.6 or pcd_1[i][j][0] > 0.6 or pcd_1[i][j][1] < -0.6 or pcd_1[i][j][1] > 0.6 or pcd_1[i][j][2] < 0.7525: #-0.5, 0.52, -0.55, 0.55
                    dilated_gt_seg_loose[i][j] = 0  
        # breakpoint()
        final_gt_seg_l = connected_components(dilated_gt_seg_loose, gripper_pos, category_name)
        

        non_zero_coords = np.nonzero(dilated_gt_seg_loose)
        
        if non_zero_coords[0].size != 0:
            # Bounding box coordinates
            y_min, x_min = np.min(non_zero_coords[0]), np.min(non_zero_coords[1])
            y_max, x_max = np.max(non_zero_coords[0]), np.max(non_zero_coords[1])
            # Width and height of the bounding box
            width = x_max - x_min + 1
            height = y_max - y_min + 1
            plt.imshow(rgb_images[0])
            plt.imshow(dilated_gt_seg_loose, alpha=0.5)
            plt.grid(False)
            plt.axis('off')
            # plt.plot([x_min, x_min], [y_min, y_max], color='red')
            # plt.plot([x_max, x_max], [y_min, y_max], color='red')
            # plt.plot([x_min, x_max], [y_min, y_min], color='red')
            # plt.plot([x_min, x_max], [y_max, y_max], color='red')
            # plt.plot(x_center, y_center, 'r*', markersize=16)
            cat = "LIGHT"
            print(f'x: {x_center}, y: {y_center}')
            os.makedirs(f'final_masks-5/segs/{category_name}', exist_ok=True) 
            os.makedirs(f'final_masks-5/segs/{cat}', exist_ok=True) 
            os.makedirs(f'final_masks-5/images/{cat}', exist_ok=True) 
            os.makedirs(f'final_masks-5/gt_bboxs/{cat}', exist_ok=True) 
            np.save(f'final_masks-5/gt_bboxs/{cat}/{val}.npy', np.array([x_min, y_min, x_max, y_max]))
            plt.savefig(f'final_masks-5/segs/{cat}/{val}.png')
            np.save(f'final_masks-5/segs/{category_name}/00001.npy', dilated_gt_seg_loose)
            plt.clf()
            plt.imsave(f'final_masks-5/segs/{category_name}/00001.png', rgb_images[0])
            val+=1
        else:
            print("something is wrong")

####################################################################################################################################################
# Exp 5 Sequence level norm followed by frame level norm, sequence level outlier removal
base_directory = '/home/jess/TaskSeg/tsg/jpgs'

# Use glob to find all directories that start with "12"
pattern = os.path.join(base_directory, 'LIGHT*')

# Iterate over each matching directory
val = 0
for dir_path in glob.glob(pattern):
    if os.path.isdir(dir_path):
        category_name = os.path.basename(dir_path)
        # breakpoint()
        rgb_images, flow_data = load_rgb_and_flow_from_directory(category_name)
        # breakpoint()
        parts = dir_path.split('-')
        start_frame = int(parts[3])
        end_frame = float(parts[4])
        frame_range = range(int(start_frame), int(end_frame + 1))

        mag_sum = np.zeros((rgb_images[0].shape[0], rgb_images[0].shape[1]))

        robot_seg = np.load(f'/home/jess/TaskSeg/tsg/robot_seg/{category_name}/00000.npy')
        robot_seg = cv2.resize(robot_seg, (400, 400), interpolation=cv2.INTER_CUBIC)
        gripper_pos = np.load(f'/home/jess/TaskSeg/tsg/gripper/{category_name}/00000.npy')
        original_size = 1280
        new_size = 400
        scaling_factor = new_size / original_size

        # Scale the gripper positions
        gripper_pos = gripper_pos * scaling_factor
        pcd_1 = np.load(f'/home/jess/TaskSeg/tsg/pcl/{category_name}/00000.npy')
        # original_size = 1280
        # new_size = 400
        # scaling_factor = new_size / original_size

        # Scale the point cloud positions
        pcd_1 = cv2.resize(pcd_1, (400, 400), interpolation=cv2.INTER_CUBIC)
        # Sequence level norm at flow, frame level norm 
        max_flow = -1000
        for idx in frame_range:
            value = idx - frame_range.start
            curr_max = get_max_flow_singular(flow_data[idx - frame_range.start])
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
            # breakpoint()
            mag_no_robot = np.multiply(magnitudes, robot_seg)
            if mag_no_robot.max() > 0:
                curr_mag = mag_no_robot/np.max(mag_no_robot)
                framemag_save_dir = os.path.join('tsg', f'seq_and_frame_norm_mag', category_name) 
                os.makedirs(framemag_save_dir, exist_ok=True) 
                plt.imsave(os.path.join(framemag_save_dir, f'{category_name}-{value}.png'), curr_mag)
                mag_sum += curr_mag

        # save the mags
        mag_save_dir = os.path.join('tsg', f'magnitudes_seq_norm_at_flow_frame_at_mag_seq_outlier_rem', category_name) 
        os.makedirs(mag_save_dir, exist_ok=True) 
        plt.imsave(os.path.join(mag_save_dir, f'{category_name}.png'), mag_sum)
        np.save(os.path.join(mag_save_dir, f'{category_name}.npy'), mag_sum)


        # Compute the average magnitude over the frame range
        norm_mag_sum = mag_sum / len(frame_range)

        data = norm_mag_sum.flatten()
        mean = np.mean(data)
        median = np.median(data)
        std = np.std(data)

        kernel = np.ones((5, 5), np.uint8)
        ground_truth_seg_loose = (norm_mag_sum >= mean + 4 * std).astype(np.uint8)
        eroded_gt_seg_loose = cv2.erode(ground_truth_seg_loose, kernel, iterations=1)
        dilated_gt_seg_loose = cv2.dilate(eroded_gt_seg_loose, kernel, iterations=1)

        seg_save_dir = os.path.join('tsg', 'final_seg-5', category_name)
        os.makedirs(seg_save_dir, exist_ok=True)
        plt.imsave(os.path.join(seg_save_dir, f'{0:05}.png'), dilated_gt_seg_loose)
        plt.clf()
        plt.figure(figsize=(16.63, 16.63))
        count = (dilated_gt_seg_loose == 1).sum()
        x_center, y_center = calculate_centroid(dilated_gt_seg_loose)#np.argwhere(dilated_gt_seg_loose==1).sum(0)/count
        for i in range(ground_truth_seg_loose.shape[0]):
            for j in range(ground_truth_seg_loose.shape[1]):
                if pcd_1[i][j][0] < -0.6 or pcd_1[i][j][0] > 0.6 or pcd_1[i][j][1] < -0.6 or pcd_1[i][j][1] > 0.6 or pcd_1[i][j][2] < 0.7525: #-0.5, 0.52, -0.55, 0.55
                    dilated_gt_seg_loose[i][j] = 0  
        # breakpoint()
        final_gt_seg_l = connected_components(dilated_gt_seg_loose, gripper_pos, category_name)
        

        non_zero_coords = np.nonzero(dilated_gt_seg_loose)

        if non_zero_coords[0].size != 0:
            # Bounding box coordinates
            y_min, x_min = np.min(non_zero_coords[0]), np.min(non_zero_coords[1])
            y_max, x_max = np.max(non_zero_coords[0]), np.max(non_zero_coords[1])
            # Width and height of the bounding box
            width = x_max - x_min + 1
            height = y_max - y_min + 1
            plt.imshow(rgb_images[0])
            plt.imshow(dilated_gt_seg_loose, alpha=0.5)
            plt.grid(False)
            plt.axis('off')
            # plt.plot([x_min, x_min], [y_min, y_max], color='red')
            # plt.plot([x_max, x_max], [y_min, y_max], color='red')
            # plt.plot([x_min, x_max], [y_min, y_min], color='red')
            # plt.plot([x_min, x_max], [y_max, y_max], color='red')
            # plt.plot(x_center, y_center, 'r*', markersize=16)
            # cat = "UMB"
            print(f'x: {x_center}, y: {y_center}')
            os.makedirs(f'final_masks-5/segs/{category_name}', exist_ok=True) 
            os.makedirs(f'final_masks-5/images/{cat}', exist_ok=True) 
            os.makedirs(f'final_masks-5/gt_bboxs/{cat}', exist_ok=True) 
            # np.save(f'final_masks-5/gt_bboxs/{cat}/{val}.npy', np.array([x_min, y_min, x_max, y_max]))
            # plt.savefig(f'final_masks-5/segs/{cat}/{val}.png')
            np.save(f'final_masks-5/segs/{category_name}/00000.npy', dilated_gt_seg_loose)
            plt.clf()
            plt.imsave(f'final_masks-5/segs/{category_name}/00000.png', rgb_images[0])
            val+=1
        else: 
            print("something went wrong part 2")
###########################################################################################################################################
# Exp 6: Frame level outlier removal, frame level norm (flow level)


# mag_sum = np.zeros((rgb_images[0].shape[0], rgb_images[0].shape[1]))
# # breakpoint()
# robot_seg = np.load(f'/home/jess/TaskSeg/tsg/robot_seg/money-same-overhead-{start_frame}-{end_frame}/00000.npy')

# ## Frame wise norm at flow
# for idx in frame_range:
#     optical_flow = get_frame_norm_flow(flow_data[idx - frame_range.start])
#     magnitudes = get_flow_magnitudes(optical_flow)
#     # breakpoint()
#     mag_no_robot = np.multiply(magnitudes, robot_seg)
#     mean = np.mean(mag_no_robot)
#     std = np.std(mag_no_robot)
#     threshold_low = mean + 8 * std
#     mag_no_robot = np.where((mag_no_robot < threshold_low), 0, mag_no_robot)
#     if mag_no_robot.max() > 0:
#         mag_sum += mag_no_robot #/np.max(mag_no_robot)

# # save the mags
# mag_save_dir = os.path.join('tsg', f'magnitudes_frame_norm_at_flow_frame_outlier_rem', category_name) 
# os.makedirs(mag_save_dir, exist_ok=True) 
# # breakpoint()
# plt.imsave(os.path.join(mag_save_dir, f'{category_name}.png'), mag_sum)
# np.save(os.path.join(mag_save_dir, f'{category_name}.npy'), mag_sum)

# # Compute the average magnitude over the frame range
# norm_mag_sum = mag_sum / len(frame_range)

# kernel = np.ones((5, 5), np.uint8)
# ground_truth_seg_loose = (norm_mag_sum >= 0.001).astype(np.uint8)
# eroded_gt_seg_loose = cv2.erode(ground_truth_seg_loose, kernel, iterations=1)
# dilated_gt_seg_loose = cv2.dilate(eroded_gt_seg_loose, kernel, iterations=1)

# seg_save_dir = os.path.join('tsg', 'final_seg-6', category_name)
# os.makedirs(seg_save_dir, exist_ok=True)
# plt.imsave(os.path.join(seg_save_dir, f'{0:05}.png'), dilated_gt_seg_loose)
# plt.clf()

# plt.imshow(rgb_images[0])
# plt.imshow(dilated_gt_seg_loose, alpha=0.5)
# os.makedirs(f'final_masks-6/{category_name}/{0:05}', exist_ok=True) 
# plt.savefig(f'final_masks-6/{category_name}/{0:05}/'+str(datetime.now())+'.png',bbox_inches='tight', pad_inches=0)
# plt.clf()


###########################################################################################################################################
# Exp 7: Frame level outlier removal, sequence level norm (flow level)

# mag_sum = np.zeros((rgb_images[0].shape[0], rgb_images[0].shape[1]))

# # Sequence level norm at flow
# max_flow = -1000
# for idx in frame_range:
#     curr_max = get_max_flow_singular(flow_data[idx - frame_range.start])
#     if curr_max > max_flow:
#         max_flow = curr_max

# for idx in frame_range:
#     optical_flow = get_seq_norm_flow(flow_data[idx - frame_range.start], max_flow)
#     magnitudes = get_flow_magnitudes(optical_flow)
#     # breakpoint()
#     mag_no_robot = np.multiply(magnitudes, robot_seg)
#     mean = np.mean(mag_no_robot)
#     std = np.std(mag_no_robot)
#     threshold_low = mean + 8 * std
#     mag_no_robot = np.where((mag_no_robot < threshold_low), 0, mag_no_robot)
#     if mag_no_robot.max() > 0:
#         mag_sum += mag_no_robot #/np.max(mag_no_robot)

# # save the mags
# mag_save_dir = os.path.join('tsg', f'magnitudes_seq_norm_at_flow_frame_outlier_rem', category_name) 
# os.makedirs(mag_save_dir, exist_ok=True) 
# plt.imsave(os.path.join(mag_save_dir, f'{category_name}.png'), mag_sum)
# np.save(os.path.join(mag_save_dir, f'{category_name}.npy'), mag_sum)


# # Compute the average magnitude over the frame range
# norm_mag_sum = mag_sum / len(frame_range)

# kernel = np.ones((5, 5), np.uint8)
# ground_truth_seg_loose = (norm_mag_sum >= 0.001).astype(np.uint8)
# eroded_gt_seg_loose = cv2.erode(ground_truth_seg_loose, kernel, iterations=1)
# dilated_gt_seg_loose = cv2.dilate(eroded_gt_seg_loose, kernel, iterations=1)

# seg_save_dir = os.path.join('tsg', 'final_seg-7', category_name)
# os.makedirs(seg_save_dir, exist_ok=True)
# plt.imsave(os.path.join(seg_save_dir, f'{0:05}.png'), dilated_gt_seg_loose)
# plt.clf()

# plt.imshow(rgb_images[0])
# plt.imshow(dilated_gt_seg_loose, alpha=0.5)
# os.makedirs(f'final_masks-7/{category_name}/{0:05}', exist_ok=True) 
# plt.savefig(f'final_masks-7/{category_name}/{0:05}/'+str(datetime.now())+'.png',bbox_inches='tight', pad_inches=0)
# plt.clf()

# ###########################################################################################################################################
# # Exp 8: Frame level outlier removal, frame (mag level) and sequence level norm (flow level)

# mag_sum = np.zeros((rgb_images[0].shape[0], rgb_images[0].shape[1]))

# # Sequence level norm at flow, frame level norm 
# max_flow = -1000
# for idx in frame_range:
#     curr_max = get_max_flow_singular(flow_data[idx - frame_range.start])
#     if curr_max > max_flow:
#         max_flow = curr_max

# for idx in frame_range:
#     optical_flow = get_seq_norm_flow(flow_data[idx - frame_range.start], max_flow)
#     magnitudes = get_flow_magnitudes(optical_flow)
#     # breakpoint()
#     mag_no_robot = np.multiply(magnitudes, robot_seg)
#     mean = np.mean(mag_no_robot)
#     std = np.std(mag_no_robot)
#     threshold_low = mean + 8 * std
#     mag_no_robot = np.where((mag_no_robot < threshold_low), 0, mag_no_robot)
#     if mag_no_robot.max() > 0:
#         mag_sum += mag_no_robot/np.max(mag_no_robot)

# # save the mags
# mag_save_dir = os.path.join('tsg', f'magnitudes_seq_norm_at_flow_frame_at_mag_frame_outlier', category_name) 
# os.makedirs(mag_save_dir, exist_ok=True) 
# plt.imsave(os.path.join(mag_save_dir, f'{category_name}.png'), mag_sum)
# np.save(os.path.join(mag_save_dir, f'{category_name}.npy'), mag_sum)


# # Compute the average magnitude over the frame range
# norm_mag_sum = mag_sum / len(frame_range)

# kernel = np.ones((5, 5), np.uint8)
# ground_truth_seg_loose = (norm_mag_sum >= 0.001).astype(np.uint8)
# eroded_gt_seg_loose = cv2.erode(ground_truth_seg_loose, kernel, iterations=1)
# dilated_gt_seg_loose = cv2.dilate(eroded_gt_seg_loose, kernel, iterations=1)

# seg_save_dir = os.path.join('tsg', 'final_seg-8', category_name)
# os.makedirs(seg_save_dir, exist_ok=True)
# plt.imsave(os.path.join(seg_save_dir, f'{0:05}.png'), dilated_gt_seg_loose)
# plt.clf()

# plt.imshow(rgb_images[0])
# plt.imshow(dilated_gt_seg_loose, alpha=0.5)
# os.makedirs(f'final_masks-8/{category_name}/{0:05}', exist_ok=True) 
# plt.savefig(f'final_masks-8/{category_name}/{0:05}/'+str(datetime.now())+'.png',bbox_inches='tight', pad_inches=0)
# plt.clf()

###########################################################################################################################################
# Exp 9: Pixel level outlier removal, frame level norm (flow level)

###########################################################################################################################################
# Exp 10: Pixel level outlier removal, sequence level norm (flow level)

###########################################################################################################################################
# Exp 11: Pixel level outlier removal, frame (mag level) and sequence level norm (flow level)