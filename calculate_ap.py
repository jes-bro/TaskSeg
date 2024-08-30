import os
import glob
import numpy as np
import cv2  # OpenCV for better resizing
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score

def calculate_metrics(gt, mask):
    TP = np.sum((gt == 1) & (mask == 1))
    TN = np.sum((gt == 0) & (mask == 0))
    FP = np.sum((gt == 0) & (mask == 1))
    FN = np.sum((gt == 1) & (mask == 0))

    # Calculate MCC
    numerator = (TP * TN) - (FP * FN)
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    
    # Check for zero denominator
    if denominator == 0:
        mcc = 0  # or np.nan or any specific value to indicate undefined
    else:
        mcc = numerator / denominator

    # Calculate Dice Coefficient
    if (2 * TP + FP + FN) == 0:
        dice = 1  # Handle the edge case where there's no positive case in gt or mask
    else:
        dice = 2 * TP / (2 * TP + FP + FN)

    return mcc, dice

def plot_masks_comparison(rgb_image, gt_resized, mask, title, alpha=0.6):
    # Create a new image to hold the comparison result with transparency (RGBA)
    comparison_image = np.zeros((gt_resized.shape[0], gt_resized.shape[1], 4), dtype=np.uint8)

    # Define colors: (R, G, B, A)
    color_fp = [255, 0, 0, 150]  # Red for False Positives
    color_fn = [0, 0, 255, 150]  # Blue for False Negatives
    color_tp = [0, 255, 0, 150]  # Green for True Positives
    color_tn = [0, 0, 1, 40]     # Transparent for True Negatives

    # Assign colors based on the conditions
    comparison_image[(gt_resized == 0) & (mask == 1)] = color_fp  # False Positive
    comparison_image[(gt_resized == 1) & (mask == 0)] = color_fn  # False Negative
    comparison_image[(gt_resized == 1) & (mask == 1)] = color_tp  # True Positive
    comparison_image[(gt_resized == 0) & (mask == 0)] = color_tn  # Transparent for True Negative

    # Overlay the masks on top of the RGB image
    plt.imshow(rgb_image)
    plt.imshow(comparison_image, alpha=alpha)  # Adjust alpha for mask transparency
    plt.title(title)
    plt.axis('off')
import os
import glob
import numpy as np
import cv2  # OpenCV for better resizing
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score

def calculate_metrics(gt, mask):
    TP = np.sum((gt == 1) & (mask == 1))
    TN = np.sum((gt == 0) & (mask == 0))
    FP = np.sum((gt == 0) & (mask == 1))
    FN = np.sum((gt == 1) & (mask == 0))

    # Calculate MCC
    numerator = (TP * TN) - (FP * FN)
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    
    # Check for zero denominator
    if denominator == 0:
        mcc = 0  # or np.nan or any specific value to indicate undefined
    else:
        mcc = numerator / denominator

    # Calculate Dice Coefficient
    if (2 * TP + FP + FN) == 0:
        dice = 1  # Handle the edge case where there's no positive case in gt or mask
    else:
        dice = 2 * TP / (2 * TP + FP + FN)

    return mcc, dice

def create_negative_mask(rgb_image, mask):
    # Create a negative mask where the non-mask area is black and mask area retains the RGB values
    negative_mask = np.zeros_like(rgb_image)
    negative_mask[mask == 1] = rgb_image[mask == 1]
    return negative_mask

def save_negative_mask(negative_mask, save_path):
    # Save the negative mask image
    negative_image = Image.fromarray(negative_mask)
    negative_image.save(save_path)

# Arrays to hold predictions and ground truth for AP calculation
flowsam_predictions = []
taskseg_predictions = []
ground_truths = []

# Lists to store metrics for calculating averages
taskseg_mcc_list = []
taskseg_dice_list = []
flowsam_mcc_list = []
flowsam_dice_list = []
highres_mcc_list = []
highres_dice_list = []
highres_aps = []
full_ground_truths = []
base_directory = '/home/jess/flowsam/masks/flowisam/nonhung'
pattern = os.path.join(base_directory, 'LIGHT*')

for dir_path in glob.glob(pattern):
    if os.path.isdir(dir_path):
        cat = os.path.basename(dir_path)
        print(cat)
        gt = np.load(f'/home/jess/TaskSeg/tsg/umbrella_seg/{cat}/00000.npy')
        flowsam_mask = np.load(f'/home/jess/flowsam/masks/flowisam/nonhung/{cat}/00000.npy') 
        flowsam_mask = (flowsam_mask > 0).astype(int)
        taskseg_mask = np.load(f'/home/jess/TaskSeg/final_masks-5/segs/{cat}/00000.npy')
        taskseg_mask_high_res = np.load(f'/home/jess/TaskSeg/final_masks-5/segs/{cat}/00001.npy')
        rgb_image = Image.open(f'/home/jess/TaskSeg/final_masks-5/segs/{cat}/00000.png')
        rgb_image = np.array(rgb_image)
        rgb_image_high_res = Image.open(f'/home/jess/TaskSeg/final_masks-5/segs/{cat}/00001.png')
        rgb_image_high_res = np.array(rgb_image_high_res)

        # Improve ground truth resizing using OpenCV with cubic interpolation
        gt_resized = cv2.resize(gt, (400, 400), interpolation=cv2.INTER_CUBIC)
        gt_resized = (gt_resized > 0.5).astype(int)  # Ensure binary ground truth
        
        # Flatten and extend the lists
        flowsam_predictions.extend(flowsam_mask.flatten())
        taskseg_predictions.extend(taskseg_mask.flatten())
        highres_aps.extend(taskseg_mask_high_res.flatten())
        ground_truths.extend(gt_resized.flatten())
        full_ground_truths.extend(gt.flatten())

        # Calculate metrics for low resolution masks
        mcc_taskseg, dice_taskseg = calculate_metrics(gt_resized, taskseg_mask)
        mcc_flowsam, dice_flowsam = calculate_metrics(gt_resized, flowsam_mask)

        # Calculate metrics for high resolution mask against original ground truth
        mcc_highres, dice_highres = calculate_metrics(gt, taskseg_mask_high_res)

        # Store metrics in lists for averaging later
        taskseg_mcc_list.append(mcc_taskseg)
        taskseg_dice_list.append(dice_taskseg)
        flowsam_mcc_list.append(mcc_flowsam)
        flowsam_dice_list.append(dice_flowsam)
        highres_mcc_list.append(mcc_highres)
        highres_dice_list.append(dice_highres)

        print(f"TMCC: {mcc_taskseg:.2f}, TDC: {dice_taskseg:.2f} FMCC: {mcc_flowsam:.2f}, FDC: {dice_flowsam:.2f} HMCC: {mcc_highres:.2f}, HDC: {dice_highres:.2f}")

        # Calculate the Average Precision (AP)
        flowsam_ap = average_precision_score(gt_resized.flatten(), flowsam_mask.flatten())
        taskseg_ap = average_precision_score(gt_resized.flatten(), taskseg_mask.flatten())
        highres_ap = average_precision_score(gt.flatten(), taskseg_mask_high_res.flatten())

        print(f'TAP: {taskseg_ap:.2f} FAP: {flowsam_ap:.2f} HAP: {highres_ap:.2f}')

        # Create and save negative masks
        negative_taskseg_mask = create_negative_mask(rgb_image, taskseg_mask)
        negative_flowsam_mask = create_negative_mask(rgb_image, flowsam_mask)
        negative_highres_mask = create_negative_mask(rgb_image_high_res,taskseg_mask_high_res)

        parent_dir = os.path.dirname(dir_path)

        # Save the negative masks in the parent directory
        save_negative_mask(negative_highres_mask, os.path.join(parent_dir, f'{cat}_highres_negative.png'))
        save_negative_mask(negative_flowsam_mask, os.path.join(parent_dir, f'{cat}_flowsam_negative.png'))
        save_negative_mask(negative_taskseg_mask, os.path.join(parent_dir, f'{cat}_taskseg_negative.png'))
        # save_negative_mask(negative_flowsam_mask, os.path.join(parent_dir, f'{cat}_flowsam_negative.png'))
        # Plot the comparisons
        plt.figure(figsize=(24, 6))

        plt.subplot(1, 4, 1)
        plt.imshow(rgb_image)
        plt.title(f'RGB Image for {cat}')
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plot_masks_comparison(rgb_image, gt_resized, flowsam_mask, f'FlowSAM Mask for {cat} (AP: {flowsam_ap:.2f})')

        plt.subplot(1, 4, 3)
        plot_masks_comparison(rgb_image, gt_resized, taskseg_mask, f'TaskSeg Mask for {cat} (AP: {taskseg_ap:.2f})')

        plt.subplot(1, 4, 4)
        plot_masks_comparison(rgb_image_high_res, gt, taskseg_mask_high_res, f'High-Res TaskSeg Mask for {cat} (AP: {highres_ap:.2f})', alpha=0.8)  # Increased alpha for visibility

        plt.show()

# Convert lists to numpy arrays
flowsam_predictions = np.array(flowsam_predictions)
taskseg_predictions = np.array(taskseg_predictions)
ground_truths = np.array(ground_truths)

# Calculate the Average Precision (AP)
flowsam_ap = average_precision_score(ground_truths, flowsam_predictions)
taskseg_ap = average_precision_score(ground_truths, taskseg_predictions)
high_res_taskseg_ap = average_precision_score(full_ground_truths, highres_aps)

print(f'flowsam_ap: {flowsam_ap:.2f}, taskseg_ap: {taskseg_ap:.2f} highres_ap: {high_res_taskseg_ap:.2f}')

# Calculate and print the average MCC and Dice for FlowSAM, TaskSeg, and High-Res TaskSeg
avg_taskseg_mcc = np.mean(taskseg_mcc_list)
avg_taskseg_dice = np.mean(taskseg_dice_list)
avg_flowsam_mcc = np.mean(flowsam_mcc_list)
avg_flowsam_dice = np.mean(flowsam_dice_list)
avg_highres_mcc = np.mean(highres_mcc_list)
avg_highres_dice = np.mean(highres_dice_list)

print(f'Average TaskSeg MCC: {avg_taskseg_mcc:.2f}, Average TaskSeg Dice: {avg_taskseg_dice:.2f}')
print(f'Average FlowSAM MCC: {avg_flowsam_mcc:.2f}, Average FlowSAM Dice: {avg_flowsam_dice:.2f}')
print(f'Average High-Res TaskSeg MCC: {avg_highres_mcc:.2f}, Average High-Res TaskSeg Dice: {avg_highres_dice:.2f}')
