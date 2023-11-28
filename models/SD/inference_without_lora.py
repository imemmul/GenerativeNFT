from diffusers import StableDiffusionPipeline, DiffusionPipeline, UNet2DConditionModel
import torch
import pandas as pd
import numpy as np
import random
import os
from PIL import Image
from functools import partial
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import functional as F

path = "/home/emir/Desktop/dev/grad_weights/sd_nft_without_lora/checkpoint-2000/unet/"

unet = UNet2DConditionModel.from_pretrained(path)

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", unet=unet)
pipe.to("cuda")
df = pd.read_csv("/home/emir/Desktop/dev/datasets/nft_dataset/metadata.csv")
rand_int = np.random.randint(len(df['text']))
prompt = str(df['text'][rand_int])
print(prompt)
# image = pipe(prompt="a panda is wearing kimono and using katana weapon", num_inference_steps=50).images[0]
image = pipe(prompt=prompt, num_inference_steps=50).images[0]
image.save("nft.png")