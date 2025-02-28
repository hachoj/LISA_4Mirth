import os
import cv2
import torch
from datasets import Dataset, DatasetDict
import random
import pandas as pd

prompts_notb_txt = "processed_med_data/chest/prompts_notb.txt"
prompts_tb_txt = "processed_med_data/chest/prompts_tb.txt"
respones_txt = "processed_med_data/chest/responses.txt"

train_images_dir = "processed_med_data/chest/train/image"
train_masks_dir = "processed_med_data/chest/train/mask"

val_images_dir = "processed_med_data/chest/val/image"
val_masks_dir = "processed_med_data/chest/val/mask"

# load the random labels and responses
with open(prompts_notb_txt, "r") as f:
    notb_prompts = [line.strip() for line in f]

with open(prompts_tb_txt, "r") as f:
    tb_prompts = [line.strip() for line in f]

with open(respones_txt, "r") as f:
    all_responses = [line.strip() for line in f]

meta_data = pd.read_csv("processed_med_data/chest/MetaData.csv")


def build_records(train=True):
    printed = False
    records = []
    if train:
        images_dir = train_images_dir
        masks_dir = train_masks_dir
        for filename in os.listdir(train_images_dir):
            base_name = os.path.splitext(filename)[0]
            if not base_name:
                continue
            
            image_path = os.path.join(images_dir, base_name + ".png")
            mask_path = os.path.join(masks_dir, base_name + "_mask.png")

            base_name_int = int(base_name)
            is_tb = meta_data.loc[meta_data["id"] == base_name_int].iloc[0]["ptb"]

            if is_tb:
                text_prompt = random.choice(tb_prompts)
            else:
                text_prompt = random.choice(notb_prompts)
            text_response = random.choice(all_responses)

            record = {
                "image_path": image_path,
                "mask_path": mask_path,
                "prompt": text_prompt,
                "response": text_response,
            }
            if (random.randint(0, 15) == 0) and not printed:
                print(record)
                printed = True
            records.append(record)
    if not train:
        images_dir = val_images_dir
        masks_dir = val_masks_dir
        for filename in os.listdir(val_images_dir):
            base_name = os.path.splitext(filename)[0]
            if not base_name:
                continue
            
            image_path = os.path.join(images_dir, base_name + ".png")
            mask_path = os.path.join(masks_dir, base_name + "_mask.png")

            base_name_int = int(base_name)
            is_tb = meta_data.loc[meta_data["id"] == base_name_int].iloc[0]["ptb"]

            if is_tb:
                text_prompt = random.choice(tb_prompts)
            else:
                text_prompt = random.choice(notb_prompts)
            text_response = random.choice(all_responses)

            record = {
                "image_path": image_path,
                "mask_path": mask_path,
                "prompt": text_prompt,
                "response": text_response,
            }
            if (random.randint(0, 15) == 0) and not printed:
                print(record)
                printed = True
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

dataset_dict = dataset_dict.map(
    process_fn,
    batch_size=16,  # Process fewer examples at once
    writer_batch_size=100  # Write smaller batches to disk
)

"""
Dataset Structure

image_path: str
mask_path: str
prompt: str
response: str
binary_mask: np.ndarray
"""
 
# Saving the processed dataset to disk
dataset_dict.save_to_disk("datasets/processed_chest_dataset")