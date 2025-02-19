import argparse
import os
import sys
import torch
import bleach
import cv2
import torch.nn.functional as F
from datasets import load_from_disk 
from PIL import Image
from transformers import (
    Trainer,
    TrainingArguments,
    default_data_collator,
    GenerationConfig,
    AutoTokenizer,
    CLIPImageProcessor,
)

import warnings
warnings.filterwarnings("ignore", message=".*_register_pytree_node.*")
warnings.filterwarnings("ignore", message=".*resume_download*")
warnings.filterwarnings("ignore", message=".*generation configuration*")

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

"""
Run script:
CUDA_VISIBLE_DEVICES=0 python fine_tune.py --version 'xinlai/LISA-13B-llama2-v1'
"""

#########################################
#     Fine-Tuning Configuration         #
#########################################

# Load preprocessed dataset from disk.
dataset_dict = load_from_disk("med_data_datasets/processed_kvasir_seg_dataset")
TRAIN_DATASET = dataset_dict["train"]
VAL_DATASET = dataset_dict["validation"]

# Training hyperparameters.
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 2e-5

#########################################
#         Argument Parsing              #
#########################################

def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA Fine-Tuning Script")
    parser.add_argument("--version", default="xinlai/LISA-13B-llama2-v1")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])
    return parser.parse_args(args)

#########################################
#               Trainer                 #
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
        
        # Compute auto-regressive text loss (cross-entropy over all tokens).
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
            # Use the weighting from the original model: 2.0 for BCE and 0.5 for Dice.
            seg_loss = 2.0 * bce_loss + 0.5 * d_loss
        else:
            seg_loss = 0.0
        
        total_loss = text_loss + seg_loss
        if return_outputs:
            return total_loss, outputs
        else:
            return total_loss

#########################################
#            Main Function              #
#########################################

def main():
    args = parse_args(sys.argv[1:])
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Initialize tokenizer and add special tokens.
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.float32  # For GH200, using fp32.
    
    # Load the model.
    model = LISAForCausalLM.from_pretrained(
        args.version,
        low_cpu_mem_usage=True,
        vision_tower=args.vision_tower,
        seg_token_idx=args.seg_token_idx,
    ).float().cuda()

    # 1. Freeze everything by default.
    for name, param in model.named_parameters():
        param.requires_grad = False

    # 2. Unfreeze only the key modules:
    #    - mask_decoder
    #    - text_hidden_fcs
    #    - lm_head
    #    - embed_tokens
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
        
        # If you want to unfreeze the prompt_encoder, you could do:
        # if "prompt_encoder" in name:
        #     param.requires_grad = True

    # 3. (Optional) If LoRA is integrated, unfreeze only the LoRA adapter parameters:
    #    for name, param in model.named_parameters():
    #        if "lora_" in name.lower():
    #            param.requires_grad = True

    # Then confirm how many parameters are trainable:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable} / {total}")

    generation_config = GenerationConfig(
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Initialize vision modules.
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    # Freeze base vision and mm_projector weights.
    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    # Initialize additional LISA modules if necessary.
    if not hasattr(model.get_model(), "initialized_lisa"):
        model.get_model().initialize_lisa_modules(model.get_model().config)
        model.get_model().initialized_lisa = True

    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    # Note: Since you don't want to rewrap, we skip get_peft_model here.
    # The model already has LoRA layers integrated, so we rely on the above freezing logic.

    model.resize_token_embeddings(len(tokenizer))
    model = model.float().cuda()
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    # Setup image processors if needed.
    clip_image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    transform = ResizeLongestSide(args.image_size)

    training_args = TrainingArguments(
        output_dir="./checkpoints/polyp_only",
        evaluation_strategy="steps",
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        logging_steps=50,
        eval_steps=200,
        save_steps=200,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=False,  # Using fp32.
        report_to="none",
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=TRAIN_DATASET,
        eval_dataset=VAL_DATASET,
        data_collator=default_data_collator,
    )

    trainer.train()

if __name__ == '__main__':
    main()
