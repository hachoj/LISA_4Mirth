import torch
import sys
import json
from transformers import (
    LlavaForConditionalGeneration,
    LlavaProcessor,
    BitsAndBytesConfig,
    CLIPImageProcessor,
    LlamaTokenizer,
    GenerationConfig,
)
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

DEBUG = True  # Toggle debug prints


def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


def load_model_and_processor():
    model_path = "LISA_Plus_7b"  # Directory with your LISA_Plus_7b files

    # Load tokenizer from model directory
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    special_tokens = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<unk>",
    }
    tokenizer.add_special_tokens(special_tokens)

    # Add extra image-specific tokens (if needed)
    new_tokens = ["<im_start>", "<im_end>", "<image>"]
    tokenizer.add_tokens(new_tokens, special_tokens=True)
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 3072

    # Create image processor configuration (you may adjust these values as needed)
    image_processor_config = {
        "do_resize": True,
        "size": {"height": 336, "width": 336},
        "do_center_crop": True,
        "crop_size": {"height": 336, "width": 336},
        "do_normalize": True,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
    }
    image_processor = CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-large-patch14", **image_processor_config
    )

    # Initialize LlavaProcessor with the image processor and tokenizer
    processor = LlavaProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        feature_extractor=image_processor,
    )

    # Set vision configuration on the processor
    vision_config = {
        "image_size": 336,
        "patch_size": 14,
        "num_channels": 3,
        "num_patches": (336 // 14) ** 2,  # Expect 576 patches
    }
    processor.config = vision_config
    processor.vision_config = vision_config
    processor.patch_size = 14

    # Load the pre-trained LISA model with quantization settings
    debug_print("Loading model from:", model_path)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quant_config,
    )
    model.resize_token_embeddings(len(tokenizer))

    # LISA is trained end-to-end to decode segmentation masks.
    # Disable the separate image token mechanism so the model uses its built-in vision encoder.
    model.config.mm_use_im_start_end = False

    debug_print("Model loaded. Model configuration:")
    debug_print(json.dumps(model.config.to_dict(), indent=2))

    return model, processor


def process_image(image_path):
    """Load image from URL or local path and convert to RGB."""
    try:
        if image_path.startswith("http"):
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_path)
        debug_print(f"Loaded image with size: {image.size}")
        return image.convert("RGB")
    except Exception as e:
        debug_print("Error loading image:", e)
        raise


def generate_response(model, processor, image, prompt):
    debug_print("Processing image and prompt...")
    inputs = processor(image, text=prompt, return_tensors="pt")

    debug_print("Processor output keys:", list(inputs.keys()))
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            debug_print(f"Input tensor '{key}' shape: {value.shape}")

    # Move tensors to model device
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)

    # Set up generation configuration
    gen_config = GenerationConfig(
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        max_length=4096,
    )
    debug_print("Model device:", model.device)
    debug_print("Generation configuration:", gen_config.to_dict())

    try:
        outputs = model.generate(
            **inputs,
            generation_config=gen_config,
            do_sample=True,
            temperature=0.7,
        )
    except Exception as e:
        debug_print("Error during generation:", e)
        debug_print("Detailed input info:")
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                debug_print(f"  {key}: shape={value.shape}, device={value.device}")
        debug_print(
            "Model config mm_use_im_start_end:", model.config.mm_use_im_start_end
        )
        raise

    debug_print("Raw output from model.generate:", outputs)

    # Decode text output using the processor's tokenizer
    if isinstance(outputs, dict):
        text_output = processor.tokenizer.decode(
            outputs["text"][0], skip_special_tokens=True
        )
        seg_mask = outputs.get("segmentation_mask", None)
    else:
        text_output = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        seg_mask = None

    debug_print("Generated text output:", text_output)
    if seg_mask is not None:
        debug_print("Segmentation mask shape:", seg_mask[0].shape)

    return text_output, seg_mask


def main():
    model, processor = load_model_and_processor()

    # Make sure the image path is valid; adjust the path if needed.
    image_path = "coco2017/test2017/000000000001.jpg"  # Replace with a valid local image path or URL
    prompt = "Highlight all the cats in the image."

    try:
        image = process_image(image_path)
    except Exception as e:
        debug_print("Failed to load image:", e)
        sys.exit(1)

    debug_print("Starting inference with prompt:", prompt)
    text_output, seg_mask = generate_response(model, processor, image, prompt)

    print("\n--- Final Output ---")
    print("Text output:")
    print(text_output)

    if seg_mask is not None:
        try:
            if isinstance(seg_mask, torch.Tensor):
                mask_to_show = seg_mask[0].cpu().numpy()
            else:
                mask_to_show = seg_mask[0]
            plt.imshow(mask_to_show, cmap="gray")
            plt.title("Segmentation Mask")
            plt.axis("off")
            plt.show()
        except Exception as e:
            print("Error displaying segmentation mask:", e)
    else:
        print("No segmentation mask returned.")


if __name__ == "__main__":
    main()
