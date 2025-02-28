import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Setup paths
image_path = "unprocessed_med_data/kvasir-seg/Kvasir-SEG/Kvasir-SEG/images"
masks_path = "unprocessed_med_data/kvasir-seg/Kvasir-SEG/Kvasir-SEG/masks"

out_image_folder_train = "processed_med_data/Kvasir-SEG/images/train"
out_image_folder_val = "processed_med_data/Kvasir-SEG/images/val"
out_mask_folder_train  = "processed_med_data/Kvasir-SEG/masks/train"
out_mask_folder_val  = "processed_med_data/Kvasir-SEG/masks/val"

train_txt = "unpreocessed_med_data/kvasir-seg/train.txt"
val_txt = "unpreocessed_med_data/kvasir-seg/val.txt"

# Ensure output directories exist
for folder in [out_image_folder_train, out_image_folder_val,
               out_mask_folder_train, out_mask_folder_val,]:
    os.makedirs(folder, exist_ok=True)

# Read train and val filenames
with open(train_txt, "r") as f:
    train_names = {line.strip() for line in f if line.strip()}

with open(val_txt, "r") as f:
    val_names = {line.strip() for line in f if line.strip()}

# Process each file in the masks directory
i = 0
for filename in os.listdir(masks_path):
    i += 1
    # Remove extension if needed to match names in txt files.
    base_name = os.path.splitext(filename)[0]
    
    if base_name in train_names:
        out_mask_folder = out_mask_folder_train
        out_image_folder = out_image_folder_train
    elif base_name in val_names:
        out_mask_folder = out_mask_folder_val
        out_image_folder = out_image_folder_val
    else:
        print(f"Skipping {filename}")
        continue  # Skip files that aren't in train or validation list

    # Build input file paths
    msk_path = os.path.join(masks_path, filename)
    img_path = os.path.join(image_path, filename)

    # Read the mask and image
    mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(img_path)
    if mask is None or img is None:
        continue  # Skip if image or mask not loaded

    # Convert mask to binary
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Save outputs
    # Save outputs with PNG format for masks
    out_mask_path = os.path.join(out_mask_folder, os.path.splitext(filename)[0] + "_mask" + ".png")
    cv2.imwrite(out_mask_path, binary_mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    out_image_path = os.path.join(out_image_folder, filename)
    cv2.imwrite(out_image_path, img)
print(i)