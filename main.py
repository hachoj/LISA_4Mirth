import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.get_device_name(0)}")
from transformers import (
    LlavaProcessor,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig,
)
from PIL import Image
import requests
from io import BytesIO


def load_model_and_tokenizer():
    """Load LISA++ model and tokenizer with optimized settings for 3060"""
    model_path = "LISA_Plus_7b"

    # Initialize processor
    processor = LlavaProcessor.from_pretrained(model_path)

    # Load model with optimized settings for 12GB VRAM
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ),
        max_memory={0: "11GB"},  # Reserve 1GB for system
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
    """Generate response from LISA++"""
    # Prepare inputs using processor
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            num_beams=3,
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
