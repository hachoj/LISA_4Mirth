import os
import cv2
import torch
from datasets import Dataset, DatasetDict
import random

prompts_txt = "processed_med_data/chase-retina/prompts.txt"
respones_txt = "processed_med_data/chase-retina/responses.txt"

train_images_dir = "processed_med_data/chase-retina/train/image"
train_masks_dir = "processed_med_data/chase-retina/train/mask"

val_images_dir = "processed_med_data/chase-retina/val/image"
val_masks_dir = "processed_med_data/chase-retina/val/mask"

# load the random labels and responses
with open(prompts_txt, "r") as f:
    all_prompts = [line.strip() for line in f]

with open(respones_txt, "r") as f:
    all_responses = [line.strip() for line in f]

def build_records(train=True):
    records = []
    if train:
        images_dir = train_images_dir
        masks_dir = train_masks_dir
        for filename in os.listdir(train_images_dir):
            base_name = os.path.splitext(filename)[0]
            if not base_name:
                continue
            
            image_path = os.path.join(images_dir, base_name + ".jpg")
            mask_path1 = os.path.join(masks_dir, base_name + "_mask1.png")
            mask_path2 = os.path.join(masks_dir, base_name + "_mask2.png")
            
            text_prompt1 = random.choice(all_prompts)
            text_response1 = random.choice(all_responses)
            text_prompt2 = random.choice(all_prompts)
            text_response2 = random.choice(all_responses)

            record = {
                "image_path": image_path,
                "mask_path": mask_path1,
                "prompt": text_prompt1,
                "response": text_response1,
            }
            records.append(record)
            record = {
                "image_path": image_path,
                "mask_path": mask_path2,
                "prompt": text_prompt2,
                "response": text_response2,
            }
            records.append(record)
    if not train:
        images_dir = val_images_dir
        masks_dir = val_masks_dir
        for filename in os.listdir(val_images_dir):
            base_name = os.path.splitext(filename)[0]
            if not base_name:
                continue
            
            image_path = os.path.join(images_dir, base_name + ".jpg")
            mask_path1 = os.path.join(masks_dir, base_name + "_mask1.png")
            mask_path2 = os.path.join(masks_dir, base_name + "_mask2.png")

            text_prompt1 = random.choice(all_prompts)
            text_response1 = random.choice(all_responses)
            text_prompt2 = random.choice(all_prompts)
            text_response2 = random.choice(all_responses)

            record = {
                "image_path": image_path,
                "mask_path": mask_path1,
                "prompt": text_prompt1,
                "response": text_response1,
            }
            records.append(record)
            record = {
                "image_path": image_path,
                "mask_path": mask_path2,
                "prompt": text_prompt2,
                "response": text_response2,
            }
            records.append(record)
    return records

train_records = build_records(train=True)
val_records = build_records(train=False)

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
dataset_dict.save_to_disk("datasets/processed_chase_retina_dataset")