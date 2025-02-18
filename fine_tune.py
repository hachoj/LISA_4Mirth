import argparse
import os
import re
import sys
import torch
import bleach
import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk 
from PIL import Image
from transformers import (
    Trainer,
    TrainingArguments,
    default_data_collator,
    GenerationConfig,
    BitsAndBytesConfig,
    AutoTokenizer,
    CLIPImageProcessor,
    GenerationConfig,
    AutoTokenizer,
)

# I had to use an old version of transformers for this model to work
# But this old version had annoying warning so I suppress them here.
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
run script:
CUDA_VISIBLE_DEVICES=0 python fine_tune.py --version 'xinlai/LISA-13B-llama2-v1' --load_in_4bit
"""

#########################################
#     Fine-Tuning Configuration         #
#########################################

# These settings are now hard-coded instead of provided via command-line.
dataset_dict = load_from_disk("med_data_datasets/processed_kvasir_seg_dataset")
TRAIN_DATASET = dataset_dict["train"]
VAL_DATASET = dataset_dict["validation"]
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
    parser.add_argument(
        "--precision",
        default="fp16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
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

# #########################################
# #         Preprocessing Function        #
# #########################################

# def preprocess(
#     x,
#     pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
#     pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
#     img_size=1024,
# ) -> torch.Tensor:
#     """Normalize pixel values and pad to a square input."""
#     # Normalize colors
#     x = (x - pixel_mean) / pixel_std
#     # Pad
#     h, w = x.shape[-2:]
#     padh = img_size - h
#     padw = img_size - w
#     x = F.pad(x, (0, padw, 0, padh))
#     return x

#########################################
#               Trainer                 #
#########################################


 def dice_loss(mask_logits, mask_labels, smooth=1e-6):
    """
    Compute Dice loss.
    Assumes mask_logits are raw outputs; applies sigmoid to convert to probabilities.
    """
    probs = torch.sigmoid(mask_logits)
    # Flatten the tensors to compute per-sample dice
    probs_flat = probs.view(probs.size(0), -1)
    labels_flat = mask_labels.view(mask_labels.size(0), -1).float()
    intersection = (probs_flat * labels_flat).sum(dim=1)
    union = probs_flat.sum(dim=1) + labels_flat.sum(dim=1)
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        print(inputs) 
        # I think this will have have binary_mask rather than mask_labels but I have to run this
        # to be able to see

        outputs = model(**inputs)  # Forward pass; should return a dict with 'logits' and optionally 'mask_logits'
        logits = outputs.logits  # Text logits (batch, seq_len, vocab_size)
        labels = inputs.get("labels")  # (batch, seq_len)

        # Compute text loss (auto-regressive, computed for every token)
        text_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="mean")

        # Compute segmentation loss if available:
        if hasattr(outputs, "mask_logits") and inputs.get("mask_labels") is not None:
            # Compute BCE loss (assuming you want to mimic BCE as in the paper)
            # Option 1: Use BCE with logits (more stable):
            bce_loss = F.binary_cross_entropy_with_logits(
                outputs.mask_logits, 
                inputs["mask_labels"].float()
            )
            # Option 2: If you want plain BCE, apply sigmoid first:
            # probs = torch.sigmoid(outputs.mask_logits)
            # bce_loss = F.binary_cross_entropy(
            #     probs,
            #     inputs["mask_labels"].float()
            # )
            # Compute Dice loss
            d_loss = dice_loss(outputs.mask_logits, inputs["mask_labels"])
            # Average the two losses equally
            seg_loss = 2.0 * bce_loss + 0.5 * d_loss
        else:
            seg_loss = 0.0

        total_loss = 1.0 * text_loss + 1.0 * seg_loss

        if return_outputs:
            return total_loss, outputs
        else:
            return total_loss

# def inference(input_str, input_image, args, model, tokenizer, clip_image_processor, transform):
#     ## filter out special chars
#     input_str = bleach.clean(input_str)

#     print("input_str: ", input_str, "input_image: ", input_image)

#     ## input valid check
#     if not re.match(r"^[A-Za-z ,.!?\'\"]+$", input_str) or len(input_str) < 1:
#         output_str = "[Error] Invalid input: ", input_str
#         # output_image = np.zeros((128, 128, 3))
#         ## error happened
#         output_image = cv2.imread("./resources/error_happened.png")[:, :, ::-1]
#         return output_image, output_str

#     # Model Inference
#     conv = conversation_lib.conv_templates[args.conv_type].copy()
#     conv.messages = []

#     prompt = input_str
#     prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
#     if args.use_mm_start_end:
#         replace_token = (
#             DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
#         )
#         prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

#     conv.append_message(conv.roles[0], prompt)
#     conv.append_message(conv.roles[1], "")
#     prompt = conv.get_prompt()

#     image_np = cv2.imread(input_image)
#     image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
#     original_size_list = [image_np.shape[:2]]

#     image_clip = (
#         clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][
#             0
#         ]
#         .unsqueeze(0)
#         .cuda()
#     )
#     if args.precision == "bf16":
#         image_clip = image_clip.bfloat16()
#     elif args.precision == "fp16":
#         image_clip = image_clip.half()
#     else:
#         image_clip = image_clip.float()

#     image = transform.apply_image(image_np)
#     resize_list = [image.shape[:2]]

#     image = (
#         preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
#         .unsqueeze(0)
#         .cuda()
#     )
#     if args.precision == "bf16":
#         image = image.bfloat16()
#     elif args.precision == "fp16":
#         image = image.half()
#     else:
#         image = image.float()

#     input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
#     input_ids = input_ids.unsqueeze(0).cuda()

#     output_ids, pred_masks = model.evaluate(
#         image_clip,
#         image,
#         input_ids,
#         resize_list,
#         original_size_list,
#         max_new_tokens=512,
#         tokenizer=tokenizer,
#     )

#     output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
    
#     # output_ids are just the token ids for the text output

#     text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
#     text_output = text_output.replace("\n", "").replace("  ", " ")
#     text_output = text_output.split("ASSISTANT: ")[-1]

    
#     # THE OUTPUT IS A BINARY MASK NOT A POLYGON

#     save_img = None
#     for i, raw_pred_mask in enumerate(pred_masks):
#         if raw_pred_mask.shape[0] == 0:
#             continue

#         raw_pred_mask = raw_pred_mask.detach().cpu().numpy()[0]
#         bin_pred_mask = raw_pred_mask > 0

#         save_img = image_np.copy()
#         save_img[bin_pred_mask] = (
#             image_np * 0.5
#             + bin_pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
#         )[bin_pred_mask]

#     output_str = "ASSITANT: " + text_output  # input_str
#     if save_img is not None:
#         output_image = save_img  # input_image
#     else:
#         ## no seg output
#         output_image = cv2.imread("./resources/no_seg_out.png")[:, :, ::-1]
#     return output_image, output_str


#########################################
#            Main Function              #
#########################################

def main():

    # input_text = input("Enter your text: ")
    # input_image = input("Enter your image path: ")
    # output_image, output_text = inference(input_text, input_image, args, model, tokenizer, clip_image_processor, transform)
    # print("Output text: ", output_text)
    # import sys
    # sys.exit(0)

    args = parse_args(sys.argv[1:])
    os.makedirs(args.vis_save_path, exist_ok=True)

    """
    Not entirely sure if I need this part
    -------------------------------------
    """
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    """
    -------------------------------------
    """

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model = LISAForCausalLM.from_pretrained(
        args.version,
        low_cpu_mem_usage=True,
        vision_tower=args.vision_tower,
        seg_token_idx=args.seg_token_idx,
        **kwargs
    )

    generation_config = GenerationConfig(
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    # Setup training arguments for Trainer.
    training_args = TrainingArguments(
        output_dir="./checkpoints/polyp_only",  # Where to save checkpoints
        evaluation_strategy="steps",
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        logging_steps=50,
        eval_steps=200,
        save_steps=200,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=(args.precision == "fp16"),
        report_to="none",
    )

    # Initialize custom Trainer.
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=TRAIN_DATASET,
        eval_dataset=VAL_DATASET,
        data_collator=default_data_collator,
        # tokenizer=tokenizer,
    )

    # Start training.
    trainer.train()  

if __name__ == '__main__':
    main()