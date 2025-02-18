from transformers import pipeline

model = pipeline("image-segmentation", model="LISA_Plus_7b")

image_path = "coco2017/test2017/000000000009.jpg"
query = "Highlight all the cats in the image."
result = model(image_path, query)
print(result)
