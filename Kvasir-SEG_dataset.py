import os
import cv2
import torch
from datasets import Dataset, DatasetDict

train_txt = "MED_DATA/Kvasir-SEG/train.txt"
val_txt = "MED_DATA/Kvasir-SEG/val.txt"

train_images_dir = "MED_DATA/Kvasir-SEG/images/train"
train_masks_dir = "MED_DATA/Kvasir-SEG/masks/train"

val_images_dir = "MED_DATA/Kvasir-SEG/images/val"
val_masks_dir = "MED_DATA/Kvasir-SEG/masks/val"

def build_records(txt_file, images_dir, masks_dir):
    records = []
    with open(txt_file, "r") as f:
        for line in f:
            filename = line.strip()
            if not filename:
                continue
            
            image_path = os.path.join(images_dir, filename + ".jpg")
            mask_path = os.path.join(masks_dir, filename + ".jpg")
            
            # Example: text prompt, or label if you have one
            text_prompt = f"Segment the brain region in {filename}"
            text_label = "Brain Tumor"  # Or your ground-truth text

            record = {
                "image_path": image_path,
                "mask_path": mask_path,
                "prompt": text_prompt,
                "text_label": text_label,
            }
            records.append(record)
    return records

train_records = build_records(train_txt, train_images_dir, train_masks_dir)
val_records = build_records(val_txt, val_images_dir, val_masks_dir)

train_dataset = Dataset.from_list(train_records)
val_dataset = Dataset.from_list(val_records)

# (Optional) create a DatasetDict for the Trainer
dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset
})

# Now you can map() over dataset_dict to load/process images, 
# tokenize text, and create the final tensors for your model.
# For example:
def process_fn(example):
    # 1. Load image
    img = cv2.imread(example["image_path"], cv2.IMREAD_COLOR)  # OpenCV uses BGR
    # 2. Load mask
    mask = cv2.imread(example["mask_path"], cv2.IMREAD_GRAYSCALE)
    # 3. Convert text to token IDs, etc.
    # ...
    return example  # Add new keys as needed

dataset_dict = dataset_dict.map(process_fn)