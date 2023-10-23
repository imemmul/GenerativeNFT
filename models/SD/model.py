import torch
model_path = "/home/emir/Desktop/dev/grad_weights/sd_nft/checkpoint-15000/pytorch_lora_weights.bin"
checkpoint = torch.load(model_path)
print(checkpoint.keys)