from transformers import ViTImageProcessor

model = "google/vit-large-patch16-384"
processor = ViTImageProcessor.from_pretrained(model)
print(processor)