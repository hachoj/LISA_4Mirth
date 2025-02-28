import os
import cv2
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Setup paths
image_path = "unprocessed_med_data/chest-tb/Chest-X-Ray/Chest-X-Ray/image"
masks_path = "unprocessed_med_data/chest-tb/Chest-X-Ray/Chest-X-Ray/mask"

out_image_folder_train = "processed_med_data/chest/train/image"
out_image_folder_val = "processed_med_data/chest/val/image"
out_mask_folder_train  = "processed_med_data/chest/train/mask"
out_mask_folder_val  = "processed_med_data/chest/val/mask"

# Ensure output directories exist
for folder in [out_image_folder_train, out_image_folder_val,
               out_mask_folder_train, out_mask_folder_val,]:
    os.makedirs(folder, exist_ok=True)

# 20 samples so 16 train and 4 val
for i, filename in enumerate(os.listdir(masks_path)):
    # Remove extension if needed to match names in txt files.
    base_name = os.path.splitext(filename)[0]
    
    # Build input file paths
    img_path = os.path.join(image_path, filename)
    msk_path = os.path.join(masks_path, filename)

    # Read the mask and image
    mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(img_path)
    if mask is None or img is None:
        print("continue")
        continue  # Skip if image or mask not loaded

    # Convert mask to binary
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Save outputs
    out_mask_folder = out_mask_folder_train if i <= 560 else out_mask_folder_val
    out_image_folder = out_image_folder_train if i <= 560 else out_image_folder_val
    # Save outputs with PNG format for masks
    out_mask_path = os.path.join(out_mask_folder, base_name + "_mask" + ".png")
    cv2.imwrite(out_mask_path, binary_mask)

    out_image_path = os.path.join(out_image_folder, base_name + ".png")
    cv2.imwrite(out_image_path, img)