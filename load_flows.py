from pathlib import Path
import numpy as np
import cv2
from skimage.transform import resize
from matplotlib import pyplot as plt

def resize_image(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

def resize_flow(flow, width, height):
    # Resize each channel separately
    flow_x, flow_y = flow[..., 0], flow[..., 1]
    flow_x_resized = cv2.resize(flow_x, (width, height), interpolation=cv2.INTER_LINEAR)
    flow_y_resized = cv2.resize(flow_y, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.stack((flow_x_resized, flow_y_resized), axis=-1)

def load_rgb_and_flow_from_directory(category_name):
    rgb_save_dir = Path('tsg') / 'jpgs' / category_name
    flow_save_dir = Path('tsg') / 'non_normalized_flow' / category_name

    target_width, target_height = 400, 400
    rgb_images = []
    flow_data = []

    for rgb_file in sorted(rgb_save_dir.glob('*.jpg')):
        flow_file = flow_save_dir / (rgb_file.stem + '.npy')
        if rgb_file.exists() and flow_file.exists():
            # Load RGB and flow data
            rgb_image = plt.imread(str(rgb_file))
            optical_flow = np.load(flow_file)

            # Resize RGB and flow data
            # resized_rgb = resize_image(rgb_image, target_width, target_height)
            # resized_flow = resize_flow(optical_flow, target_width, target_height)

            resized_rgb = cv2.resize(rgb_image, (400, 400), interpolation=cv2.INTER_CUBIC)
            rgb_images.append(resized_rgb / 255)
            resized_flow = cv2.resize(optical_flow, (400, 400), interpolation=cv2.INTER_CUBIC)
            flow_data.append(resized_flow)
        else:
            print(f"Warning: Missing data for {rgb_file.stem}")

    return rgb_images, flow_data

def load_rgb_and_flow_from_directory_no_resize(category_name):
    rgb_save_dir = Path('tsg') / 'jpgs' / category_name
    flow_save_dir = Path('tsg') / 'non_normalized_flow' / category_name

    rgb_images = []
    flow_data = []

    for rgb_file in sorted(rgb_save_dir.glob('*.jpg')):
        flow_file = flow_save_dir / (rgb_file.stem + '.npy')
        if rgb_file.exists() and flow_file.exists():
            # Load RGB and flow data
            rgb_image = plt.imread(str(rgb_file))
            optical_flow = np.load(flow_file)

            # # Resize RGB and flow data
            # resized_rgb = resize_image(rgb_image, target_width, target_height)
            # resized_flow = resize_flow(optical_flow, target_width, target_height)

            rgb_images.append(rgb_image)
            flow_data.append(optical_flow)
        else:
            print(f"Warning: Missing data for {rgb_file.stem}")

    return rgb_images, flow_data