from transformers import BlipForConditionalGeneration, AutoProcessor
import torch
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import random


device = "cuda" if torch.cuda.is_available() else "cpu"

model = BlipForConditionalGeneration.from_pretrained("dblasko/blip-dalle3-img2prompt").to(device)
processor = AutoProcessor.from_pretrained("dblasko/blip-dalle3-img2prompt")

dataset_dir = "/home/emir/Desktop/dev/datasets/nft_dataset/NFT_IMAGES"
metadata_dir =  "/home/emir/Desktop/dev/datasets/nft_dataset/metadata.csv"
output_dir = "/home/emir/Desktop/dev/GenerativeNFT/models/img2prompt/example_images"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image_files = []
collection_names = set()
selected_images = []
image_identifiers = []
prompt = []

files = os.listdir(dataset_dir)
image_extensions = ['.jpg', '.jpeg', '.png']

for file in files:
    for img in os.listdir(os.path.join(dataset_dir, file)):
        if any(img.lower().endswith(image_ext) for image_ext in image_extensions):
            image_files.append((os.path.join(dataset_dir, file, img), file))
            collection_names.add(file)
            #print(f"collection names: {collection_names}")
            image_identifiers.append(os.path.splitext(img)[0])


selected_collections = random.sample(collection_names, 5)

if "y00ts" in selected_collections:
    selected_collections.remove("y00ts")

for collection in selected_collections:
    #print(f"selected collection: {collection}")
    collection_images = [img for img, col in image_files if col == collection]
    if len(collection_images) >= 4:
        selected_images.extend(random.sample(collection_images, 4))
    else:
        selected_images.extend(collection_images)
selected_images = selected_images[:20]
file_name_for_csv = []
df = pd.read_csv(metadata_dir)
texts = []
for i, image_file in enumerate(selected_images):
    #print(f"image_file: {image_file.split('/')[-2:]}")
    #for i in image_file.split('/')[-2:]:
    image_file_splitted = image_file.split('/')[-2:]
    collection_name = image_file.split('/')[-2:][0]
    image_number = image_file.split('/')[-2:][1]
    combined_identifier = collection_name + '/' + image_number
    # print(f"Collection name: {collection_name}")
    # print(f"Image number: {image_number}")
    # print(f"Combined identifier: {combined_identifier}")
    text = df[df['file_name'] == combined_identifier]['text'].values[0]
    prompt.append(text)
    #print(text)
    #print(f"Image file splitted: {image_file_splitted}")
    image = Image.open(image_file)
    image_filename = os.path.join(output_dir, f"image_{str(i+1)}.png")
    image.save(image_filename)

    plt.figure()
    plt.imshow(image)
    plt.show()

    processed_image = processor(images=image, return_tensors="pt", padding="max_length", max_length=64, truncation=True)
    pixel_values = processed_image.pixel_values.to(device)
    captions = [text]  
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Generated caption: {generated_caption}\nReal caption: {text}")

# inputs = processor(images=image_files, text=prompt, return_tensors="pt").to(device)
# pixel_values = inputs.pixel_values
# caption = prompt

# generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
# generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# print(f"Generated caption: {generated_caption}\nReal caption: {caption}")
