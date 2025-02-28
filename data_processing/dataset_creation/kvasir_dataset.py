import os
import cv2
import torch
from datasets import Dataset, DatasetDict
import random

train_split_txt = "processed_med_data/Kvasir-SEG/train.txt"
val_split_txt = "processed_med_data/Kvasir-SEG/val.txt"
all_prompts_txt = "processed_med_data/Kvasir-SEG/random_prompts.txt"
all_respones_txt = "processed_med_data/Kvasir-SEG/random_responses.txt"

train_images_dir = "processed_med_data/Kvasir-SEG/images/train"
train_masks_dir = "processed_med_data/Kvasir-SEG/masks/train"

val_images_dir = "processed_med_data/Kvasir-SEG/images/val"
val_masks_dir = "processed_med_data/Kvasir-SEG/masks/val"

# load the random labels and responses
with open(all_prompts_txt, "r") as f:
    all_prompts = [line.strip() for line in f]

with open(all_respones_txt, "r") as f:
    all_responses = [line.strip() for line in f]

def build_records(txt_file, images_dir, masks_dir):
    records = []
    with open(txt_file, "r") as f:
        for line in f:
            filename = line.strip()
            if not filename:
                continue
            
            image_path = os.path.join(images_dir, filename + ".jpg")
            mask_path = os.path.join(masks_dir, filename + "_mask.png")
            
            text_prompt = random.choice(all_prompts)
            text_response = random.choice(all_responses)

            record = {
                "image_path": image_path,
                "mask_path": mask_path,
                "prompt": text_prompt,
                "response": text_response,
            }
            records.append(record)
    return records

train_records = build_records(train_split_txt, train_images_dir, train_masks_dir)
val_records = build_records(val_split_txt, val_images_dir, val_masks_dir)

train_dataset = Dataset.from_list(train_records)
val_dataset = Dataset.from_list(val_records)


dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset
})

def process_fn(example):
    msk_path = example["mask_path"]

    mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return example

    _, binary_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    example["binary_mask"] = binary_mask
    return example

dataset_dict = dataset_dict.map(process_fn)

"""
Dataset Structure

image_path: str
mask_path: str
prompt: str
response: str
binary_mask: np.ndarray
"""
 
# Saving the processed dataset to disk
dataset_dict.save_to_disk("datasets/processed_kvasir_seg_dataset")