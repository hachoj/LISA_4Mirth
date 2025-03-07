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

from transformers import CLIPImageProcessor
from model.segment_anything.utils.transforms import ResizeLongestSide

clip_image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
transform = ResizeLongestSide(1024)


def preprocess(x, pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1,1,1),
               pixel_std=torch.Tensor([58.395,57.12,57.375]).view(-1,1,1),
               img_size=1024):
    """Normalize and pad to a square input."""
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = torch.nn.functional.pad(x, (0, padw, 0, padh))
    return x


tokenizer = AutoTokenizer.from_pretrained(
    "xinlai/LISA-13B-llama2-v1",
    cache_dir=None,
    model_max_length=512,
    padding_side="right",
    use_fast=False,
)
tokenizer.pad_token = tokenizer.unk_token
seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

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
            
            # pick random label and response
            text_prompt = random.choice(all_prompts)
            text_response = random.choice(all_responses)
            
            prompt_tokens = tokenizer_image_token(text_prompt, tokenizer, return_tensors="pt").unsqueeze(0)
            response_tokens = tokenizer_image_token(text_response, tokenizer, return_tensors="pt").unsqueeze(0)

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