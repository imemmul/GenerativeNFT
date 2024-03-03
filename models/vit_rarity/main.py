import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

valid_list = ["azuki",
"sappy-seals",
"killabears",
"lazy-lions",
"genuine-undead",
"genesis-creepz",
"bastard-gan-punks-v2",
"pudgypenguins",
"beanzofficial",
"ninja-squad-official",
"azragames-thehopeful",
"thewarlords",
"parallel-avatars",
"pixelmongen1",
"kanpai-pandas"]

labels_dir = "/home/emir/Desktop/dev/datasets/nft_rarity_dataset/rarity_dataset/labels.csv"

def extract_rank(row):
    return row['rank'] if row and 'rank' in row else None

def convert_to_dict(string_repr):
    try:
        return ast.literal_eval(string_repr)
    except (SyntaxError, ValueError):
        return None
    


class RarityDataset(Dataset):
    def __init__(self, csv_dir, col_names, image_dir, transform):
        self.labels = pd.read_csv(csv_dir)
        self.col_names = col_names
        self.transform = transform
        self.labels['dict'] = self.labels['label'].apply(convert_to_dict)
        self.labels['rank_values'] = self.labels["dict"].apply(extract_rank)
        self.col_max_rarity = self.calculate_rarity()
        self.col_name = None
        self.drop_nan_ones()
        self.image_dir = image_dir

    def drop_nan_ones(self):
        max_col_rarity = self.col_max_rarity.copy()
        self.collection_drop = []
        for key, val in max_col_rarity.items():
            if math.isnan(val):
                print(f"{key}:{val}")
                self.col_max_rarity.pop(key)
                self.collection_drop.append(key)
        for key in self.collection_drop:
            self.labels.drop(self.labels[self.labels['data_name'].str.startswith(key)].index, inplace=True)
        self.labels.dropna(inplace=True)
        self.labels.reset_index(inplace=True)

    def __len__(self):
        return len(self.labels)

    def calculate_rarity(self):
        max_col_rarities = {}
        for col in self.col_names:
            filtered_df = self.labels[self.labels["data_name"].str.startswith(col)]
            max_col_rarities[col] = filtered_df["rank_values"].max()
        return max_col_rarities

    def get_col_labels(self, col_name):
        return self.labels[self.labels['data_name'].str.startswith(col_name)].index
    
    def __getitem__(self, index):
        self.col_name = self.labels['data_name'][index].split("_")[0] # bu olabilir
        img_dir = os.path.join(self.image_dir, self.labels['data_name'][index])
        img = np.array(Image.open(img_dir).convert('RGB'))
        if self.transform:
            img = self.transform(img)
        return img, self.labels['rank_values'][index] / self.col_max_rarity[self.col_name]
    
rarity_dataset = RarityDataset(labels_dir, valid_list, "/home/emir/Desktop/dev/datasets/nft_rarity_dataset/rarity_dataset", transform=transform)

train_size = int(0.8 * len(rarity_dataset))
test_size = len(rarity_dataset) - train_size

train_dataset, test_dataset = random_split(rarity_dataset, [train_size, test_size])

print("Training set size:", len(train_dataset))
print("Testing set size:", len(test_dataset))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
model.classifier = nn.Linear(model.config.hidden_size, 1)
optimizer = optim.SGD(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def val(model, test_loader, device):
  model.eval()
  avg_loss = 0
  with torch.no_grad():
    for img, label in test_loader:
      img, label = img.to(device), label.to(device)
      outputs = model(img)
      loss = criterion(outputs.logits.to(torch.float64).squeeze(), label)
      avg_loss += loss.item()
      print(f"validating: loss{loss.item()}")
  return avg_loss / len(test_loader)
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for i, batch in enumerate(train_loader):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        # print(outputs)
        # print(labels)
        loss = criterion(outputs.logits.to(torch.float64).squeeze(), labels)
        loss.backward()
        optimizer.step()
        print(f"iteration {i}, loss: {loss.item()}")
        train_loss += loss.item()
    val_loss = val(model, test_loader, device)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss / len(train_loader)} val_loss: {val_loss}")

torch.save(model.state_dict(), "/home/emir/Desktop/dev/datasets/nft_rarity_dataset/weights/run_01.pt")