import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification
import torch.nn as nn

class CustomDataset(Dataset):
    def __init__(self,image_dir,label_dir, transform=None):
        self.image_dir = image_dir
        self.label_df = pd.read_csv(label_dir)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224))
            
        ])
    
    
    def __len__(self):
        return len(self.label_df)
    
    def __getitem__(self,idx):
        img_name = self.label_df.iloc[idx]['data_name']
        #print(f"Image name: {img_name}")
        img_name = img_name.replace('./new_collection/', '')
        #print(f"Image name after removing first part: {img_name}")
        img_path = os.path.join(self.image_dir, img_name)
        #print(f"Image path: {img_path}")
        image = Image.open(img_path).convert('RGB')
        label = self.label_df.iloc[idx]['cls']
        #print(f"Len of csv file: {len(self.label_df)}")

        if self.transform:
            image = self.transform(image)
        
        return image, label,img_name

# 1 for rare 
# 0 for not rare
class ViTModelEvaluator:
    def __init__(self, vit_model_weights_path, device=None):
        self.vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
        self.vit_model.classifier = nn.Linear(self.vit_model.config.hidden_size, 1)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vit_model.load_state_dict(torch.load(vit_model_weights_path, map_location=self.device))
        self.vit_model.to(self.device)
        self.vit_model.eval()

    def evaluate_dataset(self, dataloader):
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for images, labels, img_names in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.vit_model(images)
                predictions = torch.sigmoid(outputs.logits)

                for prediction, label in zip(predictions, labels):
                    total_predictions += 1
                    if (prediction >= 0.5 and label == 1) or (prediction < 0.5 and label == 0):
                        correct_predictions += 1

                    if prediction >= 0.5:
                        print(f"Image {img_names} is predicted as RARE with confidence: {prediction.item()}, Actual label: {label}")
                    else:
                        print(f"Image {img_names} is predicted as NOT RARE with confidence: {1 - prediction.item()}, Actual label: {label}")

        accuracy = correct_predictions / total_predictions
        print(f"Total predictions: {total_predictions}, Correct predictions: {correct_predictions}, Accuracy: {accuracy}")