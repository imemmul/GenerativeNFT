from diffusers import StableDiffusionPipeline, DiffusionPipeline
import torch
import pandas as pd
import numpy as np
import random
import os
from PIL import Image
from functools import partial
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import functional as F

# clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

def preprocess_image(image):
    image = torch.tensor(image)
    print(image.shape)
    image = image.permute(2, 0, 1) / 255.0
    print(image.shape)
    return F.resize(image, (512, 512))

def calculate_fid(real_images, fake_images):
  fid = FrechetInceptionDistance(normalize=True)
  fid.update(real_images, real=True)
  fid.update(fake_images, real=False)
  return float(fid.compute())

# def calculate_clip_score(images, prompts):
#     images_int = (images * 255).astype("uint8")
#     clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
#     return round(float(clip_score), 4)

def dummy(images, **kwargs):
    return images, [False]
dataset_dir = "/home/emir/Desktop/dev/datasets/nft_dataset/NFT_IMAGES"
model_path = "/home/emir/Desktop/dev/grad_weights/sd_nft/checkpoint-33500"
model_path_v2 = "/home/emir/Desktop/dev/grad_weights/sd_nft_without_lora/checkpoint-2000/"
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe_2 = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe_2.unet.load_attn_procs(model_path_v2)
pipe.unet.load_attn_procs(model_path)
pipe.safety_checker = dummy
pipe_2.safety_checker = dummy
pipe.to("cuda")
pipe_2.to("cuda")
df = pd.read_csv("/home/emir/Desktop/dev/datasets/nft_dataset/metadata.csv")
prompts = []
real_imgs = []
fake_images = []
fake_images_2 = []


for _ in range(10):
    rand_int = np.random.randint(len(df['text']))
    prompt = str(df['text'][rand_int])
    img_path = os.path.join(dataset_dir, df['file_name'][rand_int])
    img = Image.open(fp=img_path).convert('RGB')
    img.save(f"./output/sample_{_}_real.png")
    img = np.array(img)
    real_imgs.append(preprocess_image(img))
    cleaned_prompt = prompt.split(',')
    random.shuffle(cleaned_prompt)
    elements_to_drop = 3

    result = cleaned_prompt[elements_to_drop:]
    result_string = ' '.join(result)
    print(f"creating image for: {result_string}")
    
    print(result_string)
    predicted_ = pipe(result_string, num_inference_steps=50, guidance_scale=7.5).images
    predicted_2 = pipe_2(result_string, num_inference_steps=50, guidance_scale=7.5).images
    fake_image = predicted_[0]
    fake_image_2 = predicted_2[0]
    fake_images.append(preprocess_image(np.array(fake_image)))
    fake_images_2.append(preprocess_image(np.array(fake_image_2)))
    fake_image.save(f"./output/sample_{_}_fake.png")
    fake_image_2.save(f"./output/sample_{_}_fake_2.png")
# sd_clip_score = calculate_clip_score(fake_images, prompts)
# print(f"CLIP score: {sd_clip_score}")
fake_images = torch.tensor(np.array(fake_images))
fake_images_2 = torch.tensor(np.array(fake_images_2))
real_images = torch.tensor(np.array(real_imgs))
print(f"FID score_1: {calculate_fid(real_images=real_images, fake_images=fake_images)}")
print(f"FID score_2: {calculate_fid(real_images=real_images, fake_images=fake_images_2)}")