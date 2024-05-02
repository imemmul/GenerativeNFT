import argparse
from evaluate_base_vit import CustomDataset, ViTModelEvaluator
import torch 
from torch.utils.data import DataLoader
from torchvision import transforms
import random

# def argument_parse():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--image_dir", type=str, help="Path to directory containing images")
#     parser.add_argument("--label_dir", type=str, help="Path to CSV file containing labels")
#     parser.add_argument("--vit_model_weights_path", type=str, help="Path to ViT model weights file")
#     #parser.add_argument("--subset_size", type=int, default=None, help="Number of samples for the subset (default: None, which means using the entire dataset)")
#     return parser.parse_args()  

# def main(image_dir, label_dir, vit_model_weights_path, subset_size=None):
#     custom_dataset = CustomDataset(image_dir, label_dir)
#     dataloader = DataLoader(custom_dataset, batch_size=8)
#     vit_evaluator = ViTModelEvaluator(vit_model_weights_path)
#     vit_evaluator.evaluate_dataset(dataloader)

# if __name__ == "__main__":
#     args = argument_parse()
#     main(args.image_dir, args.label_dir, args.vit_model_weights_path)

def argument_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_csv_path", type=str, help="Path to CSV file containing image paths")
    parser.add_argument("--label_csv_path", type=str, help="Path to CSV file containing labels")
    parser.add_argument("--img_dir", type=str, help="Path to directory containing images")
    parser.add_argument("--image_file_col", type=str, help="Name of the column containing image file paths in the image CSV file")
    parser.add_argument("--label_col", type=str, help="Name of the column containing labels in the label CSV file")
    parser.add_argument("--vit_model_weights_path", type=str, help="Path to ViT model weights file")
    return parser.parse_args()  

def main(image_csv_path, label_csv_path,img_dir,image_file_col, label_col, vit_model_weights_path):
    custom_dataset = CustomDataset(image_csv_path, label_csv_path,img_dir,image_file_col, label_col)
    dataloader = DataLoader(custom_dataset, batch_size=8)
    vit_evaluator = ViTModelEvaluator(vit_model_weights_path)
    vit_evaluator.evaluate_dataset(dataloader)

if __name__ == "__main__":
    args = argument_parse()
    main(args.image_csv_path, args.label_csv_path,args.img_dir,args.image_file_col, args.label_col, args.vit_model_weights_path)
