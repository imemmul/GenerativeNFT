import argparse
from evaluate_base_vit import CustomDataset, ViTModelEvaluator
import torch 
from torch.utils.data import DataLoader
from torchvision import transforms
import random

def argument_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, help="Path to directory containing images")
    parser.add_argument("--label_dir", type=str, help="Path to CSV file containing labels")
    parser.add_argument("--vit_model_weights_path", type=str, help="Path to ViT model weights file")
    parser.add_argument("--subset_size", type=int, default=None, help="Number of samples for the subset (default: None, which means using the entire dataset)")
    return parser.parse_args()  

def main(image_dir, label_dir, vit_model_weights_path, subset_size=None):
    custom_dataset = CustomDataset(image_dir, label_dir, subset_size=subset_size)
    if subset_size is not None:
        custom_dataset = custom_dataset[:subset_size]
    dataloader = DataLoader(custom_dataset, batch_size=8)
    vit_evaluator = ViTModelEvaluator(vit_model_weights_path)
    vit_evaluator.evaluate_dataset(dataloader)

if __name__ == "__main__":
    args = argument_parse()
    main(args.image_dir, args.label_dir, args.vit_model_weights_path, args.subset_size)
