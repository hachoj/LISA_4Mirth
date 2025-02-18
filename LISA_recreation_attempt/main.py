import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.get_device_name(0)}")
from transformers import (
    LlavaProcessor,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig,
    CLIPImageProcessor,
    LlamaTokenizer,
)
from PIL import Image
import requests
from io import BytesIO


def load_model_and_tokenizer():
    model_path = "LISA_Plus_7b"

    # Initialize tokenizer with existing special tokens
    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    # Add image-specific tokens without modifying existing ones
    new_tokens = {
        "im_start": "<im_start>",
        "im_end": "<im_end>",
        "image_token": "<image>",
    }

    # Keep existing special token mapping and add it to tokenizer
    special_tokens = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<unk>",  # Using unk as pad per special_tokens_map.json
    }
    tokenizer.add_special_tokens(special_tokens)  # Add this line

    # Add new tokens while preserving existing configuration
    num_added = tokenizer.add_tokens(
        [token for token in new_tokens.values()], special_tokens=True
    )

    # Update tokenizer configuration
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 3072

    # Create configuration for image processor
    image_processor_config = {
        "do_resize": True,
        "size": {"height": 336, "width": 336},
        "do_center_crop": True,
        "crop_size": {"height": 336, "width": 336},
        "do_normalize": True,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
    }

    # Initialize components
    image_processor = CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-large-patch14", **image_processor_config
    )

    # Create processor with consistent configuration
    processor = LlavaProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        feature_extractor=image_processor,
    )

    # Set vision configuration
    vision_config = {
        "image_size": 336,
        "patch_size": 14,
        "num_channels": 3,
        "num_patches": (336 // 14) ** 2,  # Calculate expected number of patches
    }

    processor.config = vision_config
    processor.vision_config = vision_config
    processor.patch_size = 14

    # Load model with optimized settings
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ),
    )

    # Resize token embeddings to account for new special tokens
    model.resize_token_embeddings(len(tokenizer))

    return model, processor


def process_image(image_path):
    """Load and preprocess image"""
    if image_path.startswith("http"):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path)
    return image


def generate_response(model, processor, image, prompt):
    # Process the image and text together. The processor should return a dictionary
    # that includes both the image features (e.g., "pixel_values") and the text tokens.
    inputs = processor(image, text=prompt, return_tensors="pt")
    print("Processor output keys:", inputs.keys())  # Debugging: check output keys

    # For LLaVa, if the processor doesn't automatically add image start/end tokens,
    # we manually add them. The config indicates mm_use_im_start_end is true,
    # so we expect start and end tokens to be present.
    if "image_tokens" not in inputs:
        # Retrieve the start and end token IDs. Here we use the model's bos/eos token IDs.
        # Adjust these if your model requires specific image token IDs.
        image_start_token = torch.tensor([[model.config.bos_token_id]])
        image_end_token = torch.tensor([[model.config.eos_token_id]])

        # Flatten the pixel values if needed.
        # Note: The exact shape may depend on how your processor outputs pixel features.
        flattened_pixels = inputs["pixel_values"].flatten(1)

        # Concatenate the start token, the flattened pixel values, and the end token.
        inputs["image_tokens"] = torch.cat(
            [image_start_token, flattened_pixels, image_end_token], dim=1
        )

    # Ensure that all input tensors are moved to the same device as the model.
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)

    # Generate the output using the model's generate function.
    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=0.7,
        max_length=4096,  # Adjust this value if necessary.
    )

    return outputs


def main():
    # Load model and processor
    model, processor = load_model_and_tokenizer()

    # Example usage
    image_path = "coco2017/test2017/000000000001.jpg"
    prompt = "Who are you?"

    # Process image
    image = process_image(image_path)

    # Generate response
    response = generate_response(model, processor, image, prompt)
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
