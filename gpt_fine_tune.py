import argparse
import os
import re
import sys
import time
import warnings
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from functools import partial

import transformers
from transformers import (
    Trainer,
    TrainingArguments,
    default_data_collator,
    GenerationConfig,
    BitsAndBytesConfig,
    AutoTokenizer,
    CLIPImageProcessor,
)

# Suppress annoying warnings from the old transformers version
warnings.filterwarnings("ignore", message=".*_register_pytree_node.*")
warnings.filterwarnings("ignore", message=".*resume_download*")
warnings.filterwarnings("ignore", message=".*generation configuration*")

# Import your model and helper functions
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

#############################################
#            Argument Parsing               #
#############################################

def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA Fine-Tuning & Training")
    # Core arguments (your original ones)
    parser.add_argument("--version", default="xinlai/LISA-13B-llama2-v1", help="Model version to load")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str, help="Path to save visualization outputs")
    parser.add_argument(
        "--precision",
        default="fp16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="Precision for inference/training",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="Image size")
    parser.add_argument("--model_max_length", default=512, type=int, help="Max sequence length")
    parser.add_argument("--lora_r", default=8, type=int, help="LoRA rank")
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str, help="Vision tower model")
    parser.add_argument("--local-rank", default=0, type=int, help="Local rank for distributed training")
    parser.add_argument("--load_in_8bit", action="store_true", default=False, help="Load model in 8-bit")
    parser.add_argument("--load_in_4bit", action="store_true", default=False, help="Load model in 4-bit")
    parser.add_argument("--use_mm_start_end", action="store_true", default=True, help="Use multimodal start/end tokens")
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
        help="Conversation type to use",
    )

    # Additional training arguments (from official script)
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run evaluation.")
    parser.add_argument("--train_dataset", default="", type=str, help="Name/path of the training dataset")
    parser.add_argument("--eval_dataset", default="", type=str, help="Name/path of the evaluation dataset")
    parser.add_argument("--num_train_epochs", default=3, type=int, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", default=1, type=int, help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", default=2, type=int, help="Evaluation batch size per device")
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="Learning rate")
    # (You can add more training-specific arguments as needed.)

    return parser.parse_args(args)

#############################################
#         Preprocessing Functions         #
#############################################

def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

#############################################
#          Inference Function             #
#############################################

def inference(input_str, input_image, args, model, tokenizer, clip_image_processor, transform):
    # Clean the input string (you may use bleach.clean if needed)
    input_str = re.sub(r'[^\w\s,.!?\'"]', '', input_str)
    if not re.match(r"^[A-Za-z ,.!?\'\"]+$", input_str) or len(input_str) < 1:
        output_str = "[Error] Invalid input: " + input_str
        output_image = cv2.imread("./resources/error_happened.png")[:, :, ::-1]
        return output_image, output_str

    # Prepare conversation prompt
    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + input_str
    if args.use_mm_start_end:
        replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()

    # Read and preprocess the image
    image_np = cv2.imread(input_image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]

    image_clip = (
        clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0]
        .unsqueeze(0)
        .cuda()
    )
    if args.precision == "bf16":
        image_clip = image_clip.bfloat16()
    elif args.precision == "fp16":
        image_clip = image_clip.half()
    else:
        image_clip = image_clip.float()

    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]
    image = (
        preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous(), img_size=args.image_size)
        .unsqueeze(0)
        .cuda()
    )
    if args.precision == "bf16":
        image = image.bfloat16()
    elif args.precision == "fp16":
        image = image.half()
    else:
        image = image.float()

    # Tokenize the prompt (using your custom image tokenization)
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).cuda()

    # Run model evaluation
    output_ids, pred_masks = model.evaluate(
        image_clip,
        image,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=512,
        tokenizer=tokenizer,
    )
    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
    text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    text_output = text_output.replace("\n", "").replace("  ", " ").split("ASSISTANT: ")[-1]

    # Process predicted segmentation mask (if any)
    save_img = None
    for raw_pred_mask in pred_masks:
        if raw_pred_mask.shape[0] == 0:
            continue
        raw_pred_mask = raw_pred_mask.detach().cpu().numpy()[0]
        bin_pred_mask = raw_pred_mask > 0
        save_img = image_np.copy()
        # Overlay the segmentation mask in red
        save_img[bin_pred_mask] = (
            image_np * 0.5 + bin_pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
        )[bin_pred_mask]

    output_str = "ASSISTANT: " + text_output
    if save_img is not None:
        output_image = save_img
    else:
        output_image = cv2.imread("./resources/no_seg_out.png")[:, :, ::-1]
    return output_image, output_str

#############################################
#               Main Function             #
#############################################

def main():
    args = parse_args(sys.argv[1:])
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create tokenizer and set special tokens
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    # Set the torch data type based on precision
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    # Prepare kwargs for model loading (including quantization settings)
    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update({
            "torch_dtype": torch.half,
            "load_in_4bit": True,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=["visual_model"],
            ),
        })
    elif args.load_in_8bit:
        kwargs.update({
            "torch_dtype": torch.half,
            "quantization_config": BitsAndBytesConfig(
                llm_int8_skip_modules=["visual_model"],
                load_in_8bit=True,
            ),
        })

    # Load model (your LISA model)
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

    # Initialize vision modules and move vision tower to the proper dtype/device
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

    # Prepare the image processor and transformation for inference
    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    #########################################
    #            Training Section           #
    #########################################
    if args.do_train:
        # Here you can load your custom dataset.
        # For illustration, we use Hugging Face’s load_dataset.
        # Replace "some_dataset" and "text" field with your actual dataset and fields.
        from datasets import load_dataset

        if args.train_dataset:
            dataset = load_dataset(args.train_dataset, split="train")
        else:
            dataset = load_dataset("some_dataset", split="train").select(range(1000))
        if args.eval_dataset:
            dataset_val = load_dataset(args.eval_dataset, split="validation")
        else:
            dataset_val = load_dataset("some_dataset", split="validation").select(range(200))

        # Map the tokenizer over the text field (customize as needed)
        dataset = dataset.map(lambda x: tokenizer(x["text"]), batched=True)
        dataset_val = dataset_val.map(lambda x: tokenizer(x["text"]), batched=True)
        dataset.set_format("torch", columns=["input_ids"])
        dataset_val.set_format("torch", columns=["input_ids"])

        training_args = TrainingArguments(
            output_dir="./checkpoints",
            evaluation_strategy="steps",
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            logging_steps=50,
            eval_steps=200,
            save_steps=200,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            fp16=(args.precision == "fp16"),
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset_val,
            data_collator=default_data_collator,
        )
        trainer.train()
    #########################################
    #           Inference Section           #
    #########################################
    else:
        input_text = input("Enter your text: ")
        input_image = input("Enter your image path: ")
        output_image, output_text = inference(input_text, input_image, args, model, tokenizer, clip_image_processor, transform)
        print("Output text:", output_text)
        # Optionally, save or display the output image:
        cv2.imwrite(os.path.join(args.vis_save_path, "output.png"), cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    main()
