from diffusers import StableDiffusionPipeline
import torch
import pandas as pd
import numpy as np
import random

# FIXME potential NSFW content

def dummy(images, **kwargs):
    return images, [False]

model_path = "/home/emir/Desktop/dev/grad_weights/sd_nft/checkpoint-15000"
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.unet.load_attn_procs(model_path)
pipe.safety_checker = dummy
pipe.to("cuda")
df = pd.read_csv("/home/emir/Desktop/dev/datasets/nft_dataset/metadata.csv")
for _ in range(10):
    rand_int = np.random.randint(len(df['text']))
    prompt = str(df['text'][rand_int])
    cleaned_prompt = prompt.split(',')
    random.shuffle(cleaned_prompt)
    cleaned_prompt.pop(random.randrange(len(cleaned_prompt)))
    cleaned_prompt = ' '.join([str(i) for i in cleaned_prompt]) + ", cat"
    print(f"creating image for: {cleaned_prompt}")
    image = pipe(cleaned_prompt, num_inference_steps=80, guidance_scale=7.5).images[0]
    image.save(f"./output/sample_{_}.png")