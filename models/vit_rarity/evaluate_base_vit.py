import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification
import torch.nn as nn
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class CustomDataset(Dataset):
    def __init__(self, image_csv_path, label_csv_path, image_dir, image_file_col, label_col, transform=None):
        image_df = pd.read_csv(image_csv_path)
        label_df = pd.read_csv(label_csv_path)
        self.image_paths = image_df[image_file_col].tolist()
        self.labels = label_df[label_col].tolist()
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
        ])
        #self.preprocess()
    
    
    def preprocess(self):
        self.label_df = self.label_df[~self.label_df['data_name'].str.contains('augmented')]
        self.label_df.reset_index(drop=True, inplace=True)
        self.label_df['data_name'] = self.label_df['data_name'].apply(lambda x: os.path.join(self.image_dir, x.replace('./new_collection/', '')))
    
    # def preprocess(self):
    #     self.label_df = self.label_df[~self.label_df['data_name'].str.contains('augmented')]
    #     self.label_df.reset_index(drop=True, inplace=True)
    #     self.label_df['data_name'] = self.label_df['data_name'].apply(lambda x: os.path.join(self.image_dir, x.replace('./new_collection/', '')))
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        #img_name = self.label_df.iloc[idx]['data_name']
        #label = self.label_df.iloc[idx]['cls']
        
        #image = Image.open(img_name).convert('RGB')

        img_name = self.image_paths[idx]
        img_path = os.path.join(self.image_dir, img_name)
        img_path = os.path.join('/content/drive/MyDrive', img_path)
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_name

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
    
    # def plot_confusion_matrix(self, confusion_mat):
    #     figure = plt.figure(figsize=(16, 12))
    #     sns.heatmap(confusion_mat,fmt='d', cmap='Blues')
    #     plt.tight_layout()
    #     plt.tight_layout()
    #     plt.xlabel('Predicted Label')
    #     plt.ylabel('True Label')
    #     plt.title('Confusion Matrix')
    #     plt.show()

    def evaluate_dataset(self, dataloader):
        true_labels = []
        predicted_probs = [] # for conf matrix

        with torch.no_grad():
            for images, labels, img_names in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.vit_model(images)
                predictions = torch.sigmoid(outputs.logits)

                true_labels.extend(labels.cpu().numpy()) # came from csv file 
                predicted_probs.extend(predictions.cpu().numpy()) # predictions based on sigmoid with ViT model 

                for prediction, label, img_name in zip(predictions, labels, img_names):
                    if(prediction >= 0.5):
                        print(f"Image {img_name} is predicted as RARE with confidence: {prediction.item()}, Actual label: {label}")
                    else:
                        print(f"Image {img_name} is predicted as NOT RARE with confidence: {prediction.item()}, Actual label: {label}")

        #true_labels = [1 if label == 1 else 0 for label in true_labels]
        #predicted_labels = [1 if prob >= 0.5 else 0 for prob in predicted_probs]
        true_labels = np.array(true_labels)
        predicted_probs = np.array(predicted_probs)
        predicted_labels = (predicted_probs >= 0.5).astype(int)

        f1 = f1_score(true_labels, predicted_labels)
        confusion_mat = confusion_matrix(true_labels,predicted_labels)
        #accuracy = sum([1 if true == pred else 0 for true, pred in zip(true_labels, predicted_labels)]) / len(true_labels)
        accuracy = accuracy_score(true_labels, predicted_labels)

        print(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1}")
        print(f"Confusion Matrix:\n{confusion_mat}")
        #self.plot_confusion_matrix(confusion_mat)



