from transformers import pipeline
import pandas as pd
import numpy as np
from PIL import Image
image_captioner = pipeline("image-to-text", model="/home/emir/Desktop/dev/grad_weights/captioning/runs_2/")
images_p = "/home/emir/Desktop/dev/datasets/nft_dataset/NFT_IMAGES"
df = pd.read_csv("/home/emir/Desktop/dev/datasets/nft_dataset/metadata.csv")

for _ in range(10):    
    rand_idx = np.random.randint(len(df))
    image_p = df['file_name'][rand_idx]
    real_text = df['text'][rand_idx]
    img = f"{images_p}/{image_p}"
    gen_text = image_captioner(img)
    print(f"gen_text: {gen_text} : count: {_}")
    print(f"real_text: {real_text} : count: {_}")
    Image.open(img).save(f"./img_{_}.png")