import os
import cv2
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np

# Setup paths
image_path = "unprocessed_med_data/chase-retina"

out_image_folder_train = "processed_med_data/chase-retina/train/image"
out_image_folder_val = "processed_med_data/chase-retina/val/image"
out_mask_folder_train  = "processed_med_data/chase-retina/train/mask"
out_mask_folder_val  = "processed_med_data/chase-retina/val/mask"

# Ensure output directories exist
for folder in [out_image_folder_train, out_image_folder_val,
               out_mask_folder_train, out_mask_folder_val,]:
    os.makedirs(folder, exist_ok=True)

# 14 samples so 11 train and 3 val
for i, filename in enumerate(os.listdir(image_path)):
    if i % 3 == 0:
        # Remove extension if needed to match names in txt files.
        base_name = os.path.splitext(filename)[0]
        
        # Build input file paths
        msk_path_1 = os.path.join(image_path, base_name + "_1stHO.png")
        msk_path_2 = os.path.join(image_path, base_name + "_2ndHO.png")
        img_path = os.path.join(image_path, base_name + ".jpg")

        # Read the mask and image
        mask1 = cv2.imread(msk_path_1, cv2.IMREAD_GRAYSCALE)
        mask2 = cv2.imread(msk_path_2, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(img_path)
        if mask1 is None or mask2 is None or img is None:
            continue  # Skip if image or mask not loaded

        # Convert mask to binary
        _, binary_mask1 = cv2.threshold(mask1, 127, 255, cv2.THRESH_BINARY)
        _, binary_mask2 = cv2.threshold(mask2, 127, 255, cv2.THRESH_BINARY)

        # Save outputs
        out_mask_folder = out_mask_folder_train if i <= 60 else out_mask_folder_val
        out_image_folder = out_image_folder_train if i <= 60 else out_image_folder_val
        if i <= 60:
            print(f"{base_name} is in train")
        else:
            print(f"{base_name} is in val")
        # Save outputs with PNG format for masks
        out_mask_path1 = os.path.join(out_mask_folder, base_name + "_mask1" + ".png")
        out_mask_path2 = os.path.join(out_mask_folder, base_name + "_mask2" + ".png")
        cv2.imwrite(out_mask_path1, binary_mask1)
        cv2.imwrite(out_mask_path2, binary_mask2)

        out_image_path = os.path.join(out_image_folder, base_name + ".jpg")
        cv2.imwrite(out_image_path, img)