import argparse
import os
import sys
import torch
import cv2
import torch.nn.functional as F
from torch.utils.data import DataLoader
from deepspeed.runtime.lr_schedules import WarmupDecayLR
from datasets import load_from_disk, concatenate_datasets
from PIL import Image
from transformers import (
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
import numpy as np
from utils.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)


# Define preprocess function (same as in the dataset creation file)
def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
):
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

"""
Fine-Tuning Constants/Configuration
"""

# Load preprocessed datasets from disk
dataset_dict_polyp = load_from_disk("datasets/processed_kvasir_seg_dataset")
dataset_dict_retina1 = load_from_disk("datasets/processed_chase_retina_dataset")
dataset_dict_retina2 = load_from_disk("datasets/processed_drive_retina_dataset") 
dataset_dict_chest = load_from_disk("datasets/processed_chest_dataset")

# Combine training datasets
TRAIN_DATASET = concatenate_datasets([
    dataset_dict_polyp["train"],
    dataset_dict_retina1["train"],
    dataset_dict_retina2["train"],
    dataset_dict_chest["train"]
])

# Combine validation datasets 
VAL_DATASET = concatenate_datasets([
    dataset_dict_polyp["validation"],
    dataset_dict_retina1["validation"],
    dataset_dict_retina2["validation"],
    dataset_dict_chest["validation"]
])

# Training hyperparameters.
NUM_TRAIN_EPOCHS = 2
# batch size can go north of 256 but choosing
# a smaller value for now
PER_DEVICE_TRAIN_BATCH_SIZE = 16
PER_DEVICE_EVAL_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-4

# Loss weighting
BCE_WEIGHT = 2.0
DICE_WEIGHT = 0.5
TEXT_LOSS_WEIGHT = 0.8
SEGMENTATION_LOSS_WEIGHT = 1.1

"""
Arugment Parsing
"""


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA Fine-Tuning Script")
    parser.add_argument("--version", default="xinlai/LISA-13B-llama2-v1")
    parser.add_argument("--vis_save_path", default="./checkpoints", type=str)
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--precision",
        default="fp32",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for training/inference",
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


args = parse_args(None)

"""
Tokenization
"""

# Initialize the CLIP image processor and resize transform.
clip_image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower)
transform = ResizeLongestSide(args.image_size)

tokenizer = AutoTokenizer.from_pretrained(
    args.version,
    cache_dir=None,
    model_max_length=args.model_max_length,
    use_fast=False,
    padding=False,
)
tokenizer.pad_token = tokenizer.unk_token

# Compute seg_token_idx from the tokenizer.
seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

"""
Image Processing
"""


def process_image(image_path, precision="fp32", image_size=1024):
    """Process image and ensure output matches SAM's expected format."""
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size = image_np.shape[:2]

    # Process for CLIP (image_clip)
    clip_out = clip_image_processor.preprocess(image_np, return_tensors="pt")[
        "pixel_values"
    ][0].unsqueeze(0)
    
    # Process for SAM - this handles resizing to longest side
    image_transformed = transform.apply_image(image_np)
    
    # Store intermediate size for SAM's postprocessing
    intermediate_size = image_transformed.shape[:2]  # Store H,W before any permuting
    
    # Convert to float32 and get into [B,C,H,W] format
    image_tensor = torch.from_numpy(image_transformed).float()
    image_tensor = image_tensor.permute(2, 0, 1).contiguous()
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    
    # Apply preprocessing (normalization and padding)
    image_tensor = preprocess(image_tensor)

    # Set precision
    if precision == "bf16":
        clip_out = clip_out.bfloat16()
        image_tensor = image_tensor.bfloat16()
    elif precision == "fp16":
        clip_out = clip_out.half()
        image_tensor = image_tensor.half()
    else:
        clip_out = clip_out.float()
        image_tensor = image_tensor.float()    
     
    return clip_out, image_tensor, intermediate_size, original_size

"""
Data Collator
"""


def custom_data_collator(features):
    batch = {}
    # Collect prompts and responses.
    batch["prompt"] = [f["prompt"] for f in features]
    batch["response"] = [f["response"] for f in features]

    clip_tensors = []
    image_tensors = []
    original_sizes = []
    input_ids_list = []
    attention_masks = []
    mask_labels = []
    resize_list = []
    for f in features:
        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []
        prompt = f"{DEFAULT_IM_END_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_START_TOKEN}\n{f['prompt']}{tokenizer.eos_token}{f['response']}{tokenizer.eos_token}"

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        # Debug image loading
        image_path = f["image_path"]
        if not os.path.exists(image_path):
            raise ValueError(f"Image not found: {image_path}")
   

        image_np = cv2.imread(f["image_path"])

        # Debug image loading
        if image_np is None:
            raise ValueError(f"Failed to load image: {image_path}")
         
        # Debug image shape
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        if len(image_np.shape) != 3 or image_np.shape[2] != 3:
            raise ValueError(f"Image must have 3 channels, got shape {image_np.shape}")

        clip, image, intermediate_size, original_size = process_image(f["image_path"], args.precision, args.image_size)

        # Verify processed tensors
        if image.shape[1] != 3:
            raise ValueError(f"Processed image must have 3 channels, got {image.shape}")

        image_tensors.append(image)

        resize_list.append(intermediate_size)
        original_sizes.append(original_size)

        clip_tensors.append(clip)
        tokenized = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=False,
            max_length=512        
        )
        input_ids_list.append(tokenized["input_ids"].squeeze(0))
        attention_masks.append(tokenized["attention_mask"].squeeze(0))
        # Convert binary mask (stored as list) to a tensor.
        # Ensure every example has a binary mask.
        if f.get("binary_mask") is None:
            raise ValueError(
                f"Missing binary mask for example with image path: {f['image_path']}"
            )
        # Ensure mask matches SAM's expected format [B, C, H, W]
        mask = torch.tensor(f["binary_mask"]).float()
        if len(mask.shape) == 2:  # [H, W]
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif len(mask.shape) == 3:  # [C, H, W]
            mask = mask.unsqueeze(0)  # [1, C, H, W]
        
        # Important: Don't interpolate here - let SAM handle it
        mask_labels.append(mask)


    # Text padding
    # NEW CODE: Handle padding for variable length sequences
    # Find the maximum sequence length in this batch
    max_length = max(ids.size(0) for ids in input_ids_list)

    padded_input_ids = []
    padded_attention_masks = []

    for _, (input_ids, mask) in enumerate(zip(input_ids_list, attention_masks)):
        # Calculate padding needed for this example
        padding_length = max_length - input_ids.size(0)
        
        if padding_length > 0:
            # Create padding tensor filled with pad_token_id
            padding = torch.full((padding_length,), tokenizer.pad_token_id, 
                                dtype=input_ids.dtype)
            # Pad the input_ids
            padded_input_ids.append(torch.cat([input_ids, padding], dim=0))
            
            # Create attention mask padding (0s)
            mask_padding = torch.zeros(padding_length, dtype=mask.dtype)
            # Pad the attention mask
            padded_attention_masks.append(torch.cat([mask, mask_padding], dim=0))
        else:
            # No padding needed
            padded_input_ids.append(input_ids)
            padded_attention_masks.append(mask)

    batch["image"] = torch.cat(image_tensors, dim=0)
    batch["image_clip"] = torch.cat(clip_tensors, dim=0)

    batch["input_ids"] = torch.stack(padded_input_ids)
    batch["labels"] = batch["input_ids"].clone()
    batch["attention_masks"] = torch.stack(padded_attention_masks)

    # offset not needed here
    # batch["masks_list"] = torch.cat(mask_labels, dim=0)  # [B, 1, H, W]
    batch["masks_list"] = mask_labels
    batch["label_list"] = [mask.shape[-2:] for mask in mask_labels]
    batch["resize_list"] = resize_list
    batch["original_sizes"] = original_sizes
    return batch


"""
Model Initialization
"""

torch_dtype = torch.float32
kwargs = {"torch_dtype": torch_dtype}
# Pass seg_token_idx to the model.
model = (
    LISAForCausalLM.from_pretrained(
        args.version,
        low_cpu_mem_usage=True,
        vision_tower=args.vision_tower,
        seg_token_idx=args.seg_token_idx,
        **kwargs,
    )
    .float()
    .cuda()
)
model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.ce_loss_weight = TEXT_LOSS_WEIGHT
model.dice_loss_weight = DICE_WEIGHT
model.bce_loss_weight = BCE_WEIGHT
model.seg_loss_weight = SEGMENTATION_LOSS_WEIGHT
# By default, freeze all the parameters
for name, param in model.named_parameters():
    param.requires_grad = False

# Unfreeze the parameters that LiSA trained on
# Not the base models parameters
for name, param in model.named_parameters():
    # For the mask decoder:
    if "mask_decoder" in name:
        param.requires_grad = True

    # For the text_hidden_fcs:
    if "text_hidden_fcs" in name:
        param.requires_grad = True

    # For the LM head:
    if "lm_head" in name:
        param.requires_grad = True

    # For the embedding layer:
    if "embed_tokens" in name:
        param.requires_grad = True

# Then confirm how many parameters are trainable:
number_of_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)
number_of_total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {number_of_trainable_params} / {number_of_total_params}")

# Set generation configuration.
generation_config = GenerationConfig(
    eos_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

# Initialize vision modules.
model.get_model().initialize_vision_modules(model.get_model().config)
vision_tower = model.get_model().get_vision_tower()
vision_tower.to(dtype=torch_dtype)

conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

model.resize_token_embeddings(len(tokenizer))
model = model.float().cuda()
vision_tower = model.get_model().get_vision_tower()
vision_tower.to(device=args.local_rank)

if args.precision == "bf16":
    model = model.bfloat16().cuda()
elif args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit):
    # For fp16, ensure the vision tower is in fp16.
    vision_tower = vision_tower.half().cuda()
    model.model.vision_tower = vision_tower
else:
    model = model.float().cuda()


"""
Training Function
"""


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


def train():
    lossi = []
    text_lossi = []
    mask_bce_lossi = []
    mask_dice_lossi = []
    mask_lossi = []
    vali = [1]
    train_loader = DataLoader(
        TRAIN_DATASET,
        batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_data_collator,
    )
    val_loader = DataLoader(
        VAL_DATASET,
        batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_data_collator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0)
    total_steps = (len(train_loader) // GRADIENT_ACCUMULATION_STEPS) * NUM_TRAIN_EPOCHS
    lr_scheduler = WarmupDecayLR(optimizer, warmup_num_steps=100, total_num_steps=total_steps)

    best_val_loss = float('inf')
    model.train()
    for epoch in range(NUM_TRAIN_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_TRAIN_EPOCHS}")
        epoch_loss = 0
        steps_in_epoch = 0
        for step, batch in enumerate(train_loader):

            num_epoch_steps = len(train_loader)

            # Move batched tensors to GPU.
            batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
            batch_size = len(batch["prompt"])

            offset = torch.arange(0, batch_size + 1).long().cuda()

            # # "masks_list": a list of ground-truth mask tensors (one per sample).
            # masks_list = [batch["mask_labels"][i] for i in range(batch_size)]
            # # "label_list": a list of shapes; here we use each mask’s shape.
            # label_list = [batch["mask_labels"][i].shape for i in range(batch_size)]
            # # "resize_list": for simplicity, we use (image_size, image_size) for each sample.
            # resize_list = [(args.image_size, args.image_size) for _ in range(batch_size)]
            
            # Now call the model’s forward function (using model_forward).
            outputs = model(
                images=batch["image"],              # segmentation input image
                images_clip=batch["image_clip"],      # CLIP input image
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                attention_masks=batch["attention_masks"],
                offset=offset,
                masks_list=batch["masks_list"],
                label_list=batch["label_list"],
                resize_list=batch["resize_list"],
            )

            loss = outputs["loss"]
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            current_loss = loss.item()
            lossi.append(current_loss)
            text_lossi.append(outputs["ce_loss"].item())
            mask_bce_lossi.append(outputs["mask_bce_loss"].item())
            mask_dice_lossi.append(outputs["mask_dice_loss"].item())
            mask_lossi.append(outputs["mask_loss"].item())
            epoch_loss += current_loss
            steps_in_epoch += 1

            if GRADIENT_ACCUMULATION_STEPS == 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                ce_loss = outputs["ce_loss"].item()
                mask_bce_loss = outputs["mask_bce_loss"].item()
                mask_dice_loss = outputs["mask_dice_loss"].item()
                mask_loss = outputs["mask_loss"].item()

                print(f"Epoch {epoch+1} Step {step//GRADIENT_ACCUMULATION_STEPS} | Text Loss: {ce_loss:.4f} | Mask BCE Loss: {mask_bce_loss:.4f} | Mask Dice Loss: {mask_dice_loss:.4f} | Mask Loss: {mask_loss:.4f} | Avg Loss: {current_loss:.4f}")
            else:            
                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    ce_loss = outputs["ce_loss"].item()
                    mask_bce_loss = outputs["mask_bce_loss"].item()
                    mask_dice_loss = outputs["mask_dice_loss"].item()
                    mask_loss = outputs["mask_loss"].item()

                    avg_loss = epoch_loss / steps_in_epoch
                    print(f"Epoch {epoch+1} Step {step//GRADIENT_ACCUMULATION_STEPS} | Text Loss: {ce_loss:.4f} | Mask BCE Loss: {mask_bce_loss:.4f} | Mask Dice Loss: {mask_dice_loss:.4f} | Mask Loss: {mask_loss:.4f} | Avg Loss: {avg_loss:.4f}")
            
            if num_epoch_steps // 3 == step or num_epoch_steps // 3 == 2*step or num_epoch_steps - 1 == step:
                # Validation loop
                val_loss = 0
                val_steps = 0
                print("\nRunning validation...")
                with torch.no_grad():
                    num_val_steps = len(val_loader)
                    for val_batch in val_loader:
                        # Move tensors to GPU and ensure proper batch dimension
                        val_batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in val_batch.items()}
                        
                        # Ensure proper batch dimension for all tensors
                        batch_size = val_batch["input_ids"].size(0)
                        offset = torch.arange(batch_size + 1).cuda()
                        
                        outputs = model(
                            images=val_batch["image"],
                            images_clip=val_batch["image_clip"],
                            input_ids=val_batch["input_ids"],
                            labels=val_batch["labels"],
                            attention_masks=val_batch["attention_masks"],
                            offset=offset,
                            masks_list=val_batch["masks_list"],
                            label_list=val_batch["label_list"],
                            resize_list=val_batch["resize_list"],
                        )
                        
                        val_loss += outputs["loss"].item()
                        val_steps += 1

                        print(f"Validation Step {val_steps}/{num_val_steps//3} | Loss: {outputs['loss'].item():.4f}")

                        if num_val_steps // 3 == val_steps:
                            break
                
                avg_val_loss = val_loss / (val_steps if val_steps > 0 else 1)
                vali.append(avg_val_loss)
                print(f"\nValidation Loss: {avg_val_loss:.4f}")
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    os.makedirs(args.vis_save_path, exist_ok=True)
                    save_path = os.path.join(args.vis_save_path, "best_model.pt")
                    torch.save(model.state_dict(), save_path)
                    print(f"New best model saved with validation loss: {avg_val_loss:.4f}")
        
        print(f"Epoch {epoch+1} completed.")

    # Save final model checkpoint.
    os.makedirs(args.vis_save_path, exist_ok=True)
    save_path = os.path.join(args.vis_save_path, "model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return lossi, text_lossi, mask_bce_lossi, mask_dice_lossi, mask_lossi, vali


"""
Run Training
"""

def main():
    import matplotlib.pyplot as plt
    lossi, text_lossi, mask_bce_lossi, mask_dice_lossi, mask_lossi, vali = train()
    
    # Convert lists to numpy arrays
    losses = {
        'total_loss': np.array(lossi),
        'text_loss': np.array(text_lossi), 
        'mask_bce_loss': np.array(mask_bce_lossi),
        'mask_dice_loss': np.array(mask_dice_lossi),
        'mask_loss': np.array(mask_lossi),
        'val_loss': np.array(vali)
    }
    
    # Save arrays
    np.save('training_losses.npy', losses)
    
    # Create a figure with 5 subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.delaxes(axes[2,1]) # Remove extra subplot
    
    # Plot each loss
    axes[0,0].plot(losses['total_loss'])
    axes[0,0].set_title('Total Loss')
    axes[0,0].set_xlabel('Steps')
    axes[0,0].set_ylabel('Loss')
    
    axes[0,1].plot(losses['text_loss'])
    axes[0,1].set_title('Text Loss')
    axes[0,1].set_xlabel('Steps')
    axes[0,1].set_ylabel('Loss')
    
    axes[1,0].plot(losses['mask_bce_loss'])
    axes[1,0].set_title('Mask BCE Loss')
    axes[1,0].set_xlabel('Steps')
    axes[1,0].set_ylabel('Loss')
    
    axes[1,1].plot(losses['mask_dice_loss'])
    axes[1,1].set_title('Mask Dice Loss')
    axes[1,1].set_xlabel('Steps')
    axes[1,1].set_ylabel('Loss')
    
    axes[2,0].plot(losses['mask_loss'])
    axes[2,0].set_title('Mask Loss')
    axes[2,0].set_xlabel('Steps')
    axes[2,0].set_ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('all_losses.png')


if __name__ == "__main__":
    main()
