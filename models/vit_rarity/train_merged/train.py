import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import ast
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import math
import os
import numpy as np
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
import pandas as pd
import argparse

def extract_rank(row):
    return row['rank'] if row and 'rank' in row else None

def convert_to_dict(string_repr):
    try:
        return ast.literal_eval(string_repr)
    except (SyntaxError, ValueError):
        return None
    

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class RarityDatasetMerged(Dataset):
    def __init__(self, label_dir_old, label_dir_new, image_dir_old, image_dir_new, transform):
        self.images_dir_old = image_dir_old
        self.images_dir_new = image_dir_new
        self.labels_old = pd.read_csv(label_dir_old)
        self.labels_new = pd.read_csv(label_dir_new)
        self.transform = transform
        
    def __len__(self):
        return len(self.labels_old) + len(self.labels_new)

    def __getitem__(self, index):
        if index >= len(self.labels_old):
            index = index - len(self.labels_old)
            if "augmented" in self.labels_new.iloc[index].data_name:
                img_dir = os.path.join(self.images_dir_new, self.labels_new.iloc[index].data_name)
            else:
                img_dir = os.path.join(self.images_dir_new, self.labels_new.iloc[index].data_name + ".png")
            try:
                img = np.array(Image.open(img_dir).convert('RGB'))
                print(self.labels_new.iloc[index].data_name)
                if self.transform:
                    img = self.transform(img)
                return img, self.labels_new.iloc[index].cls
            except Exception as e:
                pass
                
        else:
            if "augmented" in self.labels_old.iloc[index].onChainName:
                img_dir = os.path.join(self.images_dir_old, self.labels_old.iloc[index].onChainName)
            else:
                img_dir = os.path.join(self.images_dir_old, self.labels_old.iloc[index].onChainName + ".png")
            try:
                img = np.array(Image.open(img_dir).convert('RGB'))
                print(self.labels_old.iloc[index].onChainName)
                if self.transform:
                    img = self.transform(img)
                return img, self.labels_old.iloc[index].cls_label
            except Exception as e:
                pass


def _arg_parse():
    args = argparse.ArgumentParser()
    args.add_argument("--images_dir_old", type=str)
    args.add_argument("--images_dir_new", type=str)
    args.add_argument("--merge", action="store_true")
    args.add_argument("--split_ratio", type=float)
    args.add_argument("--label_dir_old", type=str)
    args.add_argument("--label_dir_new", type=str)
    args.add_argument("--num_epochs", type=int)
    return args.parse_args()

def val(model, test_loader, device, criterion):
    model.eval()
    avg_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            img, labels = batch
            img, labels = img.to(device), labels.to(device)
            outputs = model(img)
            labels = labels.unsqueeze(1).to(outputs.logits.dtype)
            probs = torch.sigmoid(outputs.logits)
            loss = criterion(probs, labels)
            avg_loss += loss.item()
            print(f"{i} iteration, Validating: loss {loss.item()}")
    return avg_loss / len(test_loader)


def _train():
    args = _arg_parse()
    dataset = RarityDatasetMerged(label_dir_old=args.label_dir_old, label_dir_new=args.label_dir_new,
                                  image_dir_old=args.images_dir_old, image_dir_new=args.images_dir_new, transform=transform)

    train_dataset, val_dataset = random_split(dataset, [int(len(dataset)*args.split_ratio), int(len(dataset) - int(len(dataset)*args.split_ratio))])
    print(len(train_dataset))
    print(len(val_dataset))
    
    print("Training set size:", len(train_dataset))
    print("Val set size:", len(val_dataset))
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
    model.classifier = nn.Linear(model.config.hidden_size, 1)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    for epoch in range(args.num_epochs):
        train_loss = 0
        for i, batch in enumerate(train_loader):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            labels = labels.unsqueeze(1).to(outputs.logits.dtype)

            probs = torch.sigmoid(outputs.logits)
            # print(f"probs: {probs}, shape: {probs.shape}")
            # print(f"labels: {labels}, shape: {labels.shape}")
            loss = criterion(probs, labels)
            loss.backward()
            optimizer.step()
            print(f"iteration {i}, loss: {loss.item()}")
            train_loss += loss.item()
        val_loss = val(model, val_loader, device, criterion=criterion)
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {train_loss / len(train_loader)} val_loss: {val_loss}")
    torch.save(model.state_dict(), "/home/emir/Desktop/dev/datasets/nft_rarity_dataset/weights/run_02_merged_clsf.pt")

if __name__ == "__main__":
    _train()