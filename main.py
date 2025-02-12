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
    """Load LISA++ model and tokenizer with optimized settings for 3060"""
    model_path = "LISA_Plus_7b"

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

    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    # Create processor with consistent configuration
    processor = LlavaProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        feature_extractor=image_processor,
    )

    # Set consistent patch size
    processor.config = {"image_size": 336, "patch_size": 14, "num_channels": 3}

    processor.vision_config = processor.config
    processor.patch_size = 14

    # Set patch size attribute directly
    setattr(processor, "patch_size", 14)

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
    """Generate response from LISA model"""
    # Prepare inputs
    inputs = processor(images=image, text=prompt, return_tensors="pt", padding=True)

    # Move inputs to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            num_beams=3,
            do_sample=True,
            temperature=0.7,
        )

    response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return response


def main():
    # Load model and processor
    model, processor = load_model_and_tokenizer()

    # Example usage
    image_path = "coco2017/test2017/000000000001.jpg"
    prompt = "What objects can you see in this image?"

    # Process image
    image = process_image(image_path)

    # Generate response
    response = generate_response(model, processor, image, prompt)
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
