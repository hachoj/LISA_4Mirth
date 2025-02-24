import argparse
import os
import sys
import torch
import cv2
import torch.nn.functional as F
from datasets import load_from_disk 
from PIL import Image
from transformers import (
    Trainer,
    TrainingArguments,
    GenerationConfig,
    AutoTokenizer,
    CLIPImageProcessor,
)
import warnings
warnings.filterwarnings("ignore", message=".*_register_pytree_node.*")
warnings.filterwarnings("ignore", message=".*resume_download*")
warnings.filterwarnings("ignore", message=".*generation configuration*")
warnings.filterwarnings("ignore", message=".*FutureWarning*")

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)

# Define preprocess function (same as in the dataset creation file)
def preprocess(x, pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
               pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
               img_size=1024):
    """Normalize and pad to a square input."""
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = torch.nn.functional.pad(x, (0, padw, 0, padh))
    return x

"""
Run script:
CUDA_VISIBLE_DEVICES=0 python fine_tune.py --version 'xinlai/LISA-13B-llama2-v1'
"""

#########################################
#     Fine-Tuning Configuration         #
#########################################

# Load preprocessed dataset from disk.
dataset_dict = load_from_disk("datasets/processed_kvasir_seg_dataset")
TRAIN_DATASET = dataset_dict["train"]
VAL_DATASET = dataset_dict["validation"]

# Training hyperparameters.
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 2e-5

# Loss weighting
BCE_WEIGHT = 2.0
DICE_WEIGHT = 0.5
TEXT_LOSS_WEIGHT = 1.0
SEGMENTATION_LOSS_WEIGHT = 1.0

#########################################
#         Argument Parsing              #
#########################################

def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA Fine-Tuning Script")
    parser.add_argument("--version", default="xinlai/LISA-13B-llama2-v1")
    parser.add_argument("--vis_save_path", default="./checkpoints", type=str)
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--precision", default="fp32", type=str, choices=["fp32", "bf16", "fp16"],
                        help="precision for training/inference")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])
    return parser.parse_args(args)

args = parse_args(None)

#########################################
#           Processors & Tokenizer        #
#########################################

# Initialize the CLIP image processor and resize transform.
clip_image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower)
transform = ResizeLongestSide(args.image_size)

tokenizer = AutoTokenizer.from_pretrained(
    args.version,
    cache_dir=None,
    model_max_length=args.model_max_length,
    padding_side="right",
    use_fast=False,
)
tokenizer.pad_token = tokenizer.unk_token

# Compute seg_token_idx from the tokenizer.
seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

#########################################
#      On-The-Fly Image Processing       #
#########################################

def process_image(image_path, precision="fp32", image_size=1024):
    """
    Loads an image from file and computes the CLIP tensor and segmentation tensor on the fly.
    Returns:
        clip_tensor: Tensor for CLIP model input.
        image_tensor: Preprocessed segmentation tensor.
        original_size: Original image shape (height, width).
    """
    image_np = cv2.imread(image_path)
    if image_np is None:
        raise ValueError(f"Cannot load image from path: {image_path}")
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size = image_np.shape[:2]

    # Process for CLIP (image_clip)
    clip_out = clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0)

    # Process for segmentation (image)
    image_transformed = transform.apply_image(image_np)
    image_tensor = torch.from_numpy(image_transformed).permute(2, 0, 1).contiguous().unsqueeze(0)
    image_tensor = preprocess(image_tensor)  # using the preprocess function defined above

    # Set precision.
    if precision == "bf16":
        clip_out = clip_out.bfloat16()
        image_tensor = image_tensor.bfloat16()
    elif precision == "fp16":
        clip_out = clip_out.half()
        image_tensor = image_tensor.half()
    else:
        clip_out = clip_out.float()
        image_tensor = image_tensor.float()

    return clip_out, image_tensor, original_size

#########################################
#       Custom Data Collator            #
#########################################

def custom_data_collator(features):
    """
    Collates a batch of examples. Loads images on the fly,
    tokenizes text, and converts the stored binary masks into a tensor for segmentation loss.
    """
    batch = {}
    # Collect prompts and responses.
    batch["prompt"] = [f["prompt"] for f in features]
    batch["response"] = [f["response"] for f in features]

    clip_tensors = []
    image_tensors = []
    original_sizes = []
    input_ids_list = []
    mask_labels = []
    for f in features:
        # Process image on the fly.
        clip_tensor, image_tensor, original_size = process_image(f["image_path"], precision=args.precision, image_size=args.image_size)
        clip_tensors.append(clip_tensor)
        image_tensors.append(image_tensor)
        original_sizes.append(original_size)
        # Tokenize combined conversation text.
        conversation = f"{f['prompt']} {f['response']}"
        tokenized = tokenizer_image_token(conversation, tokenizer, return_tensors="pt")
        input_ids_list.append(tokenized.squeeze(0))
        # Convert binary mask (stored as list) to a tensor.
        # Ensure every example has a binary mask.
        if f.get("binary_mask") is None:
            raise ValueError(f"Missing binary mask for example with image path: {f['image_path']}")
        mask_labels.append(torch.tensor(f["binary_mask"]))
    batch["image_clip"] = torch.cat(clip_tensors, dim=0)
    batch["image"] = torch.cat(image_tensors, dim=0)
    batch["original_size_list"] = original_sizes
    batch["input_ids"] = torch.stack(input_ids_list)
    batch["mask_labels"] = torch.stack(mask_labels)
    print(batch)
    import sys
    sys.exit(0)
    return batch

#########################################
#           Custom Trainer              #
#########################################

def dice_loss(mask_logits, mask_labels, smooth=1e-6):
    """
    Compute Dice loss.
    Applies sigmoid to convert raw mask logits to probabilities.
    """
    probs = torch.sigmoid(mask_logits)
    probs_flat = probs.view(probs.size(0), -1)
    labels_flat = mask_labels.view(mask_labels.size(0), -1).float()
    intersection = (probs_flat * labels_flat).sum(dim=1)
    union = probs_flat.sum(dim=1) + labels_flat.sum(dim=1)
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)  # Forward pass.
        logits = outputs.logits  # (batch, seq_len, vocab_size)
        labels = inputs.get("labels")  # (batch, seq_len)
        
        # Compute auto-regressive text loss.
        text_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="mean"
        )
        
        # Compute segmentation loss if available.
        if hasattr(outputs, "mask_logits") and inputs.get("mask_labels") is not None:
            bce_loss = F.binary_cross_entropy_with_logits(
                outputs.mask_logits, 
                inputs["mask_labels"].float()
            )
            d_loss = dice_loss(outputs.mask_logits, inputs["mask_labels"])
            seg_loss = BCE_WEIGHT * bce_loss + DICE_WEIGHT * d_loss
        else:
            seg_loss = 0.0
        
        total_loss = TEXT_LOSS_WEIGHT * text_loss + SEGMENTATION_LOSS_WEIGHT * seg_loss
        if return_outputs:
            return total_loss, outputs
        else:
            return total_loss

#########################################
#           Model Initialization        #
#########################################

# Pass seg_token_idx to the model.
model = LISAForCausalLM.from_pretrained(args.version, seg_token_idx=seg_token_idx)
# Set generation configuration.
generation_config = GenerationConfig(
    eos_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)
# Initialize vision modules and set proper dtype.
model.get_model().initialize_vision_modules(model.get_model().config)
vision_tower = model.get_model().get_vision_tower()

if args.precision == "bf16":
    model = model.bfloat16().cuda()
elif args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit):
    # For fp16, ensure the vision tower is in fp16.
    vision_tower = vision_tower.half().cuda()
    model.model.vision_tower = vision_tower
else:
    model = model.float().cuda()

#########################################
#           Trainer Setup               #
#########################################

training_args = TrainingArguments(
    output_dir=args.vis_save_path,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    evaluation_strategy="steps",
    save_strategy="steps",
    logging_steps=10,
    eval_steps=100,
    save_total_limit=2,
)

# Instantiate our CustomTrainer instead of Trainer.
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=TRAIN_DATASET,
    eval_dataset=VAL_DATASET,
    data_collator=custom_data_collator,
)

#########################################
#            Training Loop              #
#########################################

def main():
    trainer.train()
    trainer.save_model(args.vis_save_path)

if __name__ == "__main__":
    main()
