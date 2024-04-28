import os 
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification

class CustomDataset(Dataset):
    def __init__(self,image_dir,label_dir, transform=None):
        self.image_dir = image_dir
        self.label_df = pd.read_csv(label_dir)
        self.transform = transform
    
    def __len__(self):
        return len(self.label_df)
    
    def __getitem__(self,idx):
        img_name = self.label_df['data_name']
        image = Image.open(img_name).convert('RGB')
        label = self.label_df[idx]['cls']

        if self.transform:
            image = self.transform(image)
        
        return image, label

# 1 for rare 
# 0 for not rare
def evaluate_dataset(model,dataloader,device):
    model.eval()
    with torch.no_grad():
        for images,labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = torch.sigmoid(outputs.logits)

            for prediction, label in zip(predictions, labels):

                if prediction >= 0.5:
                    print(f"Image is predicted as RARE with confidence: {prediction.item()}, Actual label: {label}")
                else:
                    print(f"Image is predicted as NOT RARE with confidence: {1 - prediction.item()}, Actual label: {label}")
