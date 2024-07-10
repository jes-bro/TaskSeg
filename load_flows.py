import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_rgb_and_flow_from_directory(category_name):
    rgb_save_dir = Path('tsg') / 'jpgs' / category_name
    flow_save_dir = Path('tsg') / 'non_normalized_flow' / category_name

    rgb_images = []
    flow_data = []

    for rgb_file in sorted(rgb_save_dir.glob('*.jpg')):
        flow_file = flow_save_dir / (rgb_file.stem + '.npy')
        if rgb_file.exists() and flow_file.exists():
            rgb_image = plt.imread(rgb_file)
            optical_flow = np.load(flow_file)

            rgb_images.append(rgb_image)
            flow_data.append(optical_flow)
        else:
            print(f"Warning: Missing data for {rgb_file.stem}")

    return rgb_images, flow_data

# # Usage example
# category_name = 'money-same-front-81-135'
# rgb_images, flow_data = load_rgb_and_flow_from_directory(category_name)

# # Verify loaded data
# print(f"Loaded {len(rgb_images)} RGB images and {len(flow_data)} flow data arrays.")

# # Zip the lists together for corresponding RGB and flow data
# zipped_data = list(zip(rgb_images, flow_data))
# print(f"Zipped data contains {len(zipped_data)} pairs of RGB and flow data.")
