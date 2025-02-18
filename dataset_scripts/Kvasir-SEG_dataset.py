import os
import cv2
import torch
from datasets import Dataset, DatasetDict
import random

# tokenizer
from transformers import (
    AutoTokenizer,
)
from model.llava.mm_utils import tokenizer_image_token

tokenizer = AutoTokenizer.from_pretrained(
    "xinlai/LISA-13B-llama2-v1",
    cache_dir=None,
    model_max_length=512,
    padding_side="right",
    use_fast=False,
)
tokenizer.pad_token = tokenizer.unk_token
seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

train_txt = "MED_DATA/Kvasir-SEG/train.txt"
val_txt = "MED_DATA/Kvasir-SEG/val.txt"
random_prompts_txt = "MED_DATA/Kvasir-SEG/random_prompts.txt"
randm_responses_txt = "MED_DATA/Kvasir-SEG/random_responses.txt"

train_images_dir = "MED_DATA/Kvasir-SEG/images/train"
train_masks_dir = "MED_DATA/Kvasir-SEG/masks/train"

val_images_dir = "MED_DATA/Kvasir-SEG/images/val"
val_masks_dir = "MED_DATA/Kvasir-SEG/masks/val"

# load the random labels and responses
with open(random_prompts_txt, "r") as f:
    random_prompts = [line.strip() for line in f]

with open(randm_responses_txt, "r") as f:
    random_responses = [line.strip() for line in f]

def build_records(txt_file, images_dir, masks_dir):
    records = []
    with open(txt_file, "r") as f:
        for line in f:
            filename = line.strip()
            if not filename:
                continue
            
            image_path = os.path.join(images_dir, filename + ".jpg")
            mask_path = os.path.join(masks_dir, filename + "_mask.png")
            
            # pick random label and response
            text_prompt = random.choice(random_prompts)
            text_response = random.choice(random_responses)
            
            prompt_tokens = tokenizer_image_token(text_prompt, tokenizer, return_tensors="pt").unsqueeze(0)
            response_tokens = tokenizer_image_token(text_response, tokenizer, return_tensors="pt").unsqueeze(0)

            record = {
                "image_path": image_path,
                "mask_path": mask_path,
                "prompt": text_prompt,
                "response": text_response,
                "text_prompt": prompt_tokens,
                "text_response": response_tokens,
            }
            print(record) 

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

def process_fn(example):
    msk_path = example["mask_path"]
    img_path = example["image_path"]

    mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(img_path)
    if mask is None or img is None:
        return example
    
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    example["binary_mask"] = binary_mask
    example["image"] = img

    return example

dataset_dict = dataset_dict.map(process_fn)

# Saving the processed dataset to disk
dataset_dict.save_to_disk("processed_kvasir_seg_dataset")