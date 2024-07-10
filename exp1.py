'''
Frame level normalization at the flow vs frame level normalization at the magnitude
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# breakpoint()
from load_flows import load_rgb_and_flow_from_directory
from flow_viz import flow_to_image
from datetime import datetime
import cv2

start_frame = 76
end_frame = 98
cam = 'right_shoulder'
cat = 'money'
category_name = f'{cat}-same-{cam}-{start_frame}-{end_frame}'
rgb_images, flow_data = load_rgb_and_flow_from_directory(category_name)
# breakpoint()
frame_range = range(start_frame, end_frame + 1)

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

def connected_components(dilated_gt_seg, gripper_pos, image, max_distance = 0.1):
    # Use connected components to get closest segments to gripper
    num_labels, labeled_image = cv2.connectedComponents(dilated_gt_seg)
    close_segments = []
    for label in range(1, num_labels + 1):
        connected_component = np.column_stack(np.where(labeled_image == label))
        for point in connected_component:
            # Calculate the distance between the gripper and point cloud of pixel
            distance = np.linalg.norm(gripper_pos - image[0][1])
            if distance <= max_distance:
                close_segments.append(label)
                break
        # Check if the distance is within the specified maximum distanc
    # Create an empty binary mask with the same shape as the labeled image
    final_gt_seg_1 = np.zeros_like(labeled_image, dtype=np.uint8)
    # Iterate through each selected segment and include its pixels in the combined mask
    for label in close_segments:
        final_gt_seg_1[labeled_image == label] = 1
    return final_gt_seg_1

############################################################################################################################################
# Exp 0: frame wise normalization at the flow level, sequence level outlier removal

mag_sum = np.zeros((rgb_images[0].shape[0], rgb_images[0].shape[1]))
# breakpoint()
robot_seg = np.load(f'/home/jess/TaskSeg/tsg/robot_seg/{cat}-same-{cam}-{start_frame}-{end_frame}/00000.npy')

## Frame wise norm at flow
for idx in frame_range:
    optical_flow = get_frame_norm_flow(flow_data[idx - frame_range.start])
    magnitudes = get_flow_magnitudes(optical_flow)
    # breakpoint()
    mag_no_robot = np.multiply(magnitudes, robot_seg)
    if mag_no_robot.max() > 0:
        mag_sum += mag_no_robot #/np.max(mag_no_robot)

# save the mags
mag_save_dir = os.path.join('tsg', f'magnitudes_frame_norm_at_flow_seq_outlier_rem', category_name) 
os.makedirs(mag_save_dir, exist_ok=True) 
# breakpoint()
plt.imsave(os.path.join(mag_save_dir, f'{category_name}.png'), mag_sum)
np.save(os.path.join(mag_save_dir, f'{category_name}.npy'), mag_sum)

# Compute the average magnitude over the frame range
norm_mag_sum = mag_sum / len(frame_range)

data = norm_mag_sum.flatten()
mean = np.mean(data)
median =np.median(data)
std = np.std(data)

# Filter values above mean + n * std (where n ranges from 1 to 10)
for n in range(1, 11):
    threshold = mean + n * std
    filtered_mag_sum = np.copy(norm_mag_sum)
    filtered_mag_sum[filtered_mag_sum < threshold] = np.nan

    # Plot the filtered magnitude sum image
    plt.figure(figsize=(10, 6))
    plt.imshow(filtered_mag_sum, cmap='viridis')
    plt.colorbar()
    plt.title(f'Filtered Magnitude Sum (values >= mean + {n} * std)')
    plt.axis('off')
    plt.savefig(os.path.join(mag_save_dir, f'{category_name}-{n}.png'))

cmap = plt.get_cmap('viridis')

# Normalize the data for color mapping
norm = mcolors.Normalize(vmin=np.min(data), vmax=np.max(data))

# Create the histogram data
n, bins = np.histogram(data, bins=100)

# Scatter plot for the histogram
fig, ax = plt.subplots(figsize=(10, 6))

# Create a scatter plot of the data points with log scale
# Calculate the bin centers
bin_centers = 0.5 * (bins[:-1] + bins[1:])
bin_counts, _ = np.histogram(data, bins=bins)

# Create a scatter plot of the histogram
sc = ax.scatter(bin_centers, bin_counts, c=bin_centers, cmap=cmap, norm=norm, edgecolor='black', alpha=0.6)

# Set y-axis to log scale
ax.set_yscale('log')

# Add colorbar to the scatter plot
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Magnitude')

# Add mean and median lines
plt.axvline(mean, color='r', linestyle='dashed', linewidth=1, label='Mean')
plt.axvline(median, color='g', linestyle='dashed', linewidth=1, label='Median')


# Bootstrapping for the standard error of the median
n_bootstrap_samples = 1000
bootstrapped_medians = np.empty(n_bootstrap_samples)

for i in range(n_bootstrap_samples):
    bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
    bootstrapped_medians[i] = np.median(bootstrap_sample)

se_median = np.std(bootstrapped_medians)

# Add standard deviation lines from +1 STD to +5 STD
for i in range(1, 11):
    plt.axvline(mean + i * std, color='r', linestyle='dotted', linewidth=1, label=f'Mean + {i} STD' if i == 1 else "")
    plt.axvline(median + i * se_median, color='g', linestyle='dotted', linewidth=1, label=f'Median + {i} SE' if i == 1 else "")

plt.legend()
plt.title('Histogram of Magnitudes')
plt.xlabel('Magnitude')
plt.ylabel('Frequency (Log Scale)')
plt.grid(True, which='both', linestyle='--')

# Save the histogram
hist_save_dir = os.path.join('tsg', 'hist', category_name)
os.makedirs(hist_save_dir, exist_ok=True)
plt.savefig(os.path.join(hist_save_dir, f'{0:05}.png'))

kernel = np.ones((5, 5), np.uint8)
ground_truth_seg_loose = (norm_mag_sum >= mean + 8 * std).astype(np.uint8)
eroded_gt_seg_loose = cv2.erode(ground_truth_seg_loose, kernel, iterations=1)
dilated_gt_seg_loose = cv2.dilate(eroded_gt_seg_loose, kernel, iterations=1)

seg_save_dir = os.path.join('tsg', 'final_seg', category_name)
os.makedirs(seg_save_dir, exist_ok=True)
plt.imsave(os.path.join(seg_save_dir, f'{0:05}.png'), dilated_gt_seg_loose)
plt.clf()

plt.imshow(rgb_images[0])
plt.imshow(dilated_gt_seg_loose, alpha=0.5)
os.makedirs(f'final_masks/{category_name}/{0:05}', exist_ok=True) 
plt.savefig(f'final_masks/{category_name}/{0:05}/'+str(datetime.now())+'.png',bbox_inches='tight', pad_inches=0)
plt.clf()

##############################################################################################################################################
## EXP 2 framewise norm at magnitude agg, sequence level outlier removal
mag_sum = np.zeros((rgb_images[0].shape[0], rgb_images[0].shape[1]))

# Frame wise norm at magnitude agg
for idx in frame_range:
    optical_flow = flow_data[idx - frame_range.start]
    magnitudes = get_flow_magnitudes(optical_flow)
    mag_no_robot = np.multiply(magnitudes, robot_seg)
    if mag_no_robot.max() > 0:
        mag_sum += mag_no_robot / np.max(mag_no_robot)

# save the mags
mag_save_dir = os.path.join('tsg', f'magnitudes_frame_norm_at_mag_seq_outlier_rem', category_name) 
os.makedirs(mag_save_dir, exist_ok=True) 
plt.imsave(os.path.join(mag_save_dir, f'{category_name}.png'), mag_sum)
np.save(os.path.join(mag_save_dir, f'{category_name}.npy'), mag_sum)


# Compute the average magnitude over the frame range
norm_mag_sum = mag_sum / len(frame_range)

data = norm_mag_sum.flatten()
mean = np.mean(data)
median =np.median(data)
std = np.std(data)

# Filter values above mean + n * std (where n ranges from 1 to 10)
for n in range(1, 11):
    threshold = mean + n * std
    filtered_mag_sum = np.copy(norm_mag_sum)
    filtered_mag_sum[filtered_mag_sum < threshold] = np.nan

    # Plot the filtered magnitude sum image
    plt.figure(figsize=(10, 6))
    plt.imshow(filtered_mag_sum, cmap='viridis')
    plt.colorbar()
    plt.title(f'Filtered Magnitude Sum (values >= mean + {n} * std)')
    plt.axis('off')
    plt.savefig(os.path.join(mag_save_dir, f'{category_name}-{n}.png'))

cmap = plt.get_cmap('viridis')

# Normalize the data for color mapping
norm = mcolors.Normalize(vmin=np.min(data), vmax=np.max(data))

# Create the histogram data
n, bins = np.histogram(data, bins=100)

# Scatter plot for the histogram
fig, ax = plt.subplots(figsize=(10, 6))

# Create a scatter plot of the data points with log scale
# Calculate the bin centers
bin_centers = 0.5 * (bins[:-1] + bins[1:])
bin_counts, _ = np.histogram(data, bins=bins)

# Create a scatter plot of the histogram
sc = ax.scatter(bin_centers, bin_counts, c=bin_centers, cmap=cmap, norm=norm, edgecolor='black', alpha=0.6)

# Set y-axis to log scale
ax.set_yscale('log')

# Add colorbar to the scatter plot
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Magnitude')

# Add mean and median lines
plt.axvline(mean, color='r', linestyle='dashed', linewidth=1, label='Mean')
plt.axvline(median, color='g', linestyle='dashed', linewidth=1, label='Median')

# Add standard deviation lines from +1 STD to +5 STD
for i in range(1, 11):
    plt.axvline(mean + i * std, color='r', linestyle='dotted', linewidth=1, label=f'Mean + {i} STD' if i == 1 else "")
    plt.axvline(median + i * std, color='g', linestyle='dotted', linewidth=1, label=f'Median + {i} STD' if i == 1 else "")

plt.legend()
plt.title('Histogram of Magnitudes')
plt.xlabel('Magnitude')
plt.ylabel('Frequency')
plt.grid(True, which='both', linestyle='--')

# Save the histogram
hist_save_dir = os.path.join('tsg', 'hist-2', category_name)
os.makedirs(hist_save_dir, exist_ok=True)
plt.savefig(os.path.join(hist_save_dir, f'{0:05}.png'))

kernel = np.ones((5, 5), np.uint8)
ground_truth_seg_loose = (norm_mag_sum >= mean + 8 * std).astype(np.uint8)
eroded_gt_seg_loose = cv2.erode(ground_truth_seg_loose, kernel, iterations=1)
dilated_gt_seg_loose = cv2.dilate(eroded_gt_seg_loose, kernel, iterations=1)

seg_save_dir = os.path.join('tsg', 'final_seg-2', category_name)
os.makedirs(seg_save_dir, exist_ok=True)
plt.imsave(os.path.join(seg_save_dir, f'{0:05}.png'), dilated_gt_seg_loose)
plt.clf()

plt.imshow(rgb_images[0])
plt.imshow(dilated_gt_seg_loose, alpha=0.5)
os.makedirs(f'final_masks-2/{category_name}/{0:05}', exist_ok=True) 
plt.savefig(f'final_masks-2/{category_name}/{0:05}/'+str(datetime.now())+'.png',bbox_inches='tight', pad_inches=0)
plt.clf()



###############################################################################################################################################
# Exp 3 sequence level norm at flow level, sequence level norm

mag_sum = np.zeros((rgb_images[0].shape[0], rgb_images[0].shape[1]))

# Sequence level norm at flow
max_flow = -1000
for idx in frame_range:
    curr_max = get_max_flow_singular(flow_data[idx - frame_range.start])
    if curr_max > max_flow:
        max_flow = curr_max

for idx in frame_range:
    optical_flow = get_seq_norm_flow(flow_data[idx - frame_range.start], max_flow)
    magnitudes = get_flow_magnitudes(optical_flow)
    # breakpoint()
    mag_no_robot = np.multiply(magnitudes, robot_seg)
    if mag_no_robot.max() > 0:
        mag_sum += mag_no_robot #/np.max(mag_no_robot)

# save the mags
mag_save_dir = os.path.join('tsg', f'magnitudes_seq_norm_at_flow_seq_outlier_rem', category_name) 
os.makedirs(mag_save_dir, exist_ok=True) 
plt.imsave(os.path.join(mag_save_dir, f'{category_name}.png'), mag_sum)
np.save(os.path.join(mag_save_dir, f'{category_name}.npy'), mag_sum)


# Compute the average magnitude over the frame range
norm_mag_sum = mag_sum / len(frame_range)

data = norm_mag_sum.flatten()
mean = np.mean(data)
median =np.median(data)
std = np.std(data)

# Filter values above mean + n * std (where n ranges from 1 to 10)
for n in range(1, 11):
    threshold = mean + n * std
    filtered_mag_sum = np.copy(norm_mag_sum)
    filtered_mag_sum[filtered_mag_sum < threshold] = np.nan

    # Plot the filtered magnitude sum image
    plt.figure(figsize=(10, 6))
    plt.imshow(filtered_mag_sum, cmap='viridis')
    plt.colorbar()
    plt.title(f'Filtered Magnitude Sum (values >= mean + {n} * std)')
    plt.axis('off')
    plt.savefig(os.path.join(mag_save_dir, f'{category_name}-{n}.png'))

cmap = plt.get_cmap('viridis')

# Normalize the data for color mapping
norm = mcolors.Normalize(vmin=np.min(data), vmax=np.max(data))

# Create the histogram data
n, bins = np.histogram(data, bins=100)

# Scatter plot for the histogram
fig, ax = plt.subplots(figsize=(10, 6))

# Create a scatter plot of the data points with log scale
# Calculate the bin centers
bin_centers = 0.5 * (bins[:-1] + bins[1:])
bin_counts, _ = np.histogram(data, bins=bins)

# Create a scatter plot of the histogram
sc = ax.scatter(bin_centers, bin_counts, c=bin_centers, cmap=cmap, norm=norm, edgecolor='black', alpha=0.6)

# Set y-axis to log scale
ax.set_yscale('log')

# Add colorbar to the scatter plot
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Magnitude')

# Add mean and median lines
plt.axvline(mean, color='r', linestyle='dashed', linewidth=1, label='Mean')
plt.axvline(median, color='g', linestyle='dashed', linewidth=1, label='Median')

# Add standard deviation lines from +1 STD to +5 STD
for i in range(1, 11):
    plt.axvline(mean + i * std, color='r', linestyle='dotted', linewidth=1, label=f'Mean + {i} STD' if i == 1 else "")
    plt.axvline(median + i * std, color='g', linestyle='dotted', linewidth=1, label=f'Median + {i} STD' if i == 1 else "")

plt.legend()
plt.title('Histogram of Magnitudes')
plt.xlabel('Magnitude')
plt.ylabel('Frequency')
plt.grid(True, which='both', linestyle='--')

# Save the histogram
hist_save_dir = os.path.join('tsg', 'hist-3', category_name)
os.makedirs(hist_save_dir, exist_ok=True)
plt.savefig(os.path.join(hist_save_dir, f'{0:05}.png'))

kernel = np.ones((5, 5), np.uint8)
ground_truth_seg_loose = (norm_mag_sum >= mean + 8 * std).astype(np.uint8)
eroded_gt_seg_loose = cv2.erode(ground_truth_seg_loose, kernel, iterations=1)
dilated_gt_seg_loose = cv2.dilate(eroded_gt_seg_loose, kernel, iterations=1)

seg_save_dir = os.path.join('tsg', 'final_seg-3', category_name)
os.makedirs(seg_save_dir, exist_ok=True)
plt.imsave(os.path.join(seg_save_dir, f'{0:05}.png'), dilated_gt_seg_loose)
plt.clf()

plt.imshow(rgb_images[0])
plt.imshow(dilated_gt_seg_loose, alpha=0.5)
os.makedirs(f'final_masks-3/{category_name}/{0:05}', exist_ok=True) 
plt.savefig(f'final_masks-3/{category_name}/{0:05}/'+str(datetime.now())+'.png',bbox_inches='tight', pad_inches=0)
plt.clf()

############################################################################################################################################
# Exp 4 sequence level norm at mag level, sequence level outlier removal

mag_sum = np.zeros((rgb_images[0].shape[0], rgb_images[0].shape[1]))

# Sequence level norm at magnitude
max_flow = -1000
for idx in frame_range:
    curr_max = get_max_flow_singular(flow_data[idx - frame_range.start])
    if curr_max > max_flow:
        max_flow = curr_max

for idx in frame_range:
    optical_flow = flow_data[idx - frame_range.start]
    magnitudes = get_flow_magnitudes(optical_flow)
    # breakpoint()
    mag_no_robot = np.multiply(magnitudes, robot_seg)
    if mag_no_robot.max() > 0:
        mag_sum += mag_no_robot / max_flow

# save the mags
mag_save_dir = os.path.join('tsg', f'magnitudes_seq_norm_at_mag_seq_outlier_rem', category_name) 
os.makedirs(mag_save_dir, exist_ok=True) 
plt.imsave(os.path.join(mag_save_dir, f'{category_name}.png'), mag_sum)
np.save(os.path.join(mag_save_dir, f'{category_name}.npy'), mag_sum)


# Compute the average magnitude over the frame range
norm_mag_sum = mag_sum / len(frame_range)

data = norm_mag_sum.flatten()
mean = np.mean(data)
median =np.median(data)
std = np.std(data)

# Filter values above mean + n * std (where n ranges from 1 to 10)
for n in range(1, 11):
    threshold = mean + n * std
    filtered_mag_sum = np.copy(norm_mag_sum)
    filtered_mag_sum[filtered_mag_sum < threshold] = np.nan

    # Plot the filtered magnitude sum image
    plt.figure(figsize=(10, 6))
    plt.imshow(filtered_mag_sum, cmap='viridis')
    plt.colorbar()
    plt.title(f'Filtered Magnitude Sum (values >= mean + {n} * std)')
    plt.axis('off')
    plt.savefig(os.path.join(mag_save_dir, f'{category_name}-{n}.png'))

cmap = plt.get_cmap('viridis')

# Normalize the data for color mapping
norm = mcolors.Normalize(vmin=np.min(data), vmax=np.max(data))

# Create the histogram data
n, bins = np.histogram(data, bins=100)

# Scatter plot for the histogram
fig, ax = plt.subplots(figsize=(10, 6))

# Create a scatter plot of the data points with log scale
# Calculate the bin centers
bin_centers = 0.5 * (bins[:-1] + bins[1:])
bin_counts, _ = np.histogram(data, bins=bins)

# Create a scatter plot of the histogram
sc = ax.scatter(bin_centers, bin_counts, c=bin_centers, cmap=cmap, norm=norm, edgecolor='black', alpha=0.6)

# Set y-axis to log scale
ax.set_yscale('log')

# Add colorbar to the scatter plot
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Magnitude')

# Add mean and median lines
plt.axvline(mean, color='r', linestyle='dashed', linewidth=1, label='Mean')
plt.axvline(median, color='g', linestyle='dashed', linewidth=1, label='Median')

# Add standard deviation lines from +1 STD to +5 STD
for i in range(1, 11):
    plt.axvline(mean + i * std, color='r', linestyle='dotted', linewidth=1, label=f'Mean + {i} STD' if i == 1 else "")
    plt.axvline(median + i * std, color='g', linestyle='dotted', linewidth=1, label=f'Median + {i} STD' if i == 1 else "")

plt.legend()
plt.title('Histogram of Magnitudes')
plt.xlabel('Magnitude')
plt.ylabel('Frequency')
plt.grid(True, which='both', linestyle='--')

# Save the histogram
hist_save_dir = os.path.join('tsg', 'hist-4', category_name)
os.makedirs(hist_save_dir, exist_ok=True)
plt.savefig(os.path.join(hist_save_dir, f'{0:05}.png'))

kernel = np.ones((5, 5), np.uint8)
ground_truth_seg_loose = (norm_mag_sum >= mean + 8 * std).astype(np.uint8)
eroded_gt_seg_loose = cv2.erode(ground_truth_seg_loose, kernel, iterations=1)
dilated_gt_seg_loose = cv2.dilate(eroded_gt_seg_loose, kernel, iterations=1)

seg_save_dir = os.path.join('tsg', 'final_seg-4', category_name)
os.makedirs(seg_save_dir, exist_ok=True)
plt.imsave(os.path.join(seg_save_dir, f'{0:05}.png'), dilated_gt_seg_loose)
plt.clf()

plt.imshow(rgb_images[0])
plt.imshow(dilated_gt_seg_loose, alpha=0.5)
os.makedirs(f'final_masks-4/{category_name}/{0:05}', exist_ok=True) 
plt.savefig(f'final_masks-4/{category_name}/{0:05}/'+str(datetime.now())+'.png',bbox_inches='tight', pad_inches=0)
plt.clf()

####################################################################################################################################################
# Exp 5 Sequence level norm followed by frame level norm, sequence level outlier removal

mag_sum = np.zeros((rgb_images[0].shape[0], rgb_images[0].shape[1]))

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

# Filter values above mean + n * std (where n ranges from 1 to 10)
for n in range(1, 11):
    threshold = mean + n * std
    filtered_mag_sum = np.copy(norm_mag_sum)
    filtered_mag_sum[filtered_mag_sum < threshold] = np.nan

    # Plot the filtered magnitude sum image
    plt.figure(figsize=(10, 6))
    plt.imshow(filtered_mag_sum, cmap='viridis')
    plt.colorbar()
    plt.title(f'Filtered Magnitude Sum (values >= mean + {n} * std)')
    plt.axis('off')
    plt.savefig(os.path.join(mag_save_dir, f'{category_name}-{n}.png'))

cmap = plt.get_cmap('viridis')

# Normalize the data for color mapping
norm = mcolors.Normalize(vmin=np.min(data), vmax=np.max(data))

# Create the histogram data
n, bins = np.histogram(data, bins=100)

# Scatter plot for the histogram
fig, ax = plt.subplots(figsize=(10, 6))

# Create a scatter plot of the data points with log scale
# Calculate the bin centers
bin_centers = 0.5 * (bins[:-1] + bins[1:])
bin_counts, _ = np.histogram(data, bins=bins)

# Create a scatter plot of the histogram
sc = ax.scatter(bin_centers, bin_counts, c=bin_centers, cmap=cmap, norm=norm, edgecolor='black', alpha=0.6)

# Set y-axis to log scale
ax.set_yscale('log')

# Add colorbar to the scatter plot
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Magnitude')

# Add mean and median lines
plt.axvline(mean, color='r', linestyle='dashed', linewidth=1, label='Mean')
plt.axvline(median, color='g', linestyle='dashed', linewidth=1, label='Median')

# Add standard deviation lines from +1 STD to +5 STD
for i in range(1, 11):
    plt.axvline(mean + i * std, color='r', linestyle='dotted', linewidth=1, label=f'Mean + {i} STD' if i == 1 else "")
    plt.axvline(median + i * std, color='g', linestyle='dotted', linewidth=1, label=f'Median + {i} STD' if i == 1 else "")

plt.legend()
plt.title('Histogram of Magnitudes')
plt.xlabel('Magnitude')
plt.ylabel('Frequency (Log scale)')
plt.grid(True, which='both', linestyle='--')

# Save the histogram
hist_save_dir = os.path.join('tsg', 'hist-5', category_name)
os.makedirs(hist_save_dir, exist_ok=True)
plt.savefig(os.path.join(hist_save_dir, f'{0:05}.png'))

kernel = np.ones((5, 5), np.uint8)
ground_truth_seg_loose = (norm_mag_sum >= mean + 4 * std).astype(np.uint8)
eroded_gt_seg_loose = cv2.erode(ground_truth_seg_loose, kernel, iterations=1)
dilated_gt_seg_loose = cv2.dilate(eroded_gt_seg_loose, kernel, iterations=1)

seg_save_dir = os.path.join('tsg', 'final_seg-5', category_name)
os.makedirs(seg_save_dir, exist_ok=True)
plt.imsave(os.path.join(seg_save_dir, f'{0:05}.png'), dilated_gt_seg_loose)
plt.clf()

plt.imshow(rgb_images[0])
plt.imshow(dilated_gt_seg_loose, alpha=0.5)
os.makedirs(f'final_masks-5/{category_name}/{0:05}', exist_ok=True) 
plt.savefig(f'final_masks-5/{category_name}/{0:05}/'+str(datetime.now())+'.png',bbox_inches='tight', pad_inches=0)
plt.clf()

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