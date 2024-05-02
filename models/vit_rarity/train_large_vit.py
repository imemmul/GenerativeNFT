import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
import torch
import seaborn as sns
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class RarityDataset(Dataset):
    def __init__(self, label_dir_old, label_dir_new, image_dir_old, image_dir_new, file_path_dir_old,file_name_column,transform):
        self.images_dir_old = image_dir_old
        self.images_dir_new = image_dir_new
        #self.images_dir_augmented = image_dir_augmented
        self.labels_old = pd.read_csv(label_dir_old)
        self.labels_new = pd.read_csv(label_dir_new)
        self.file_path_dir_old = pd.read_csv(file_path_dir_old)
        self.image_paths = self.file_path_dir_old[file_name_column].tolist()
        self.file_path_dir_old[file_name_column] = '/content/drive/MyDrive' + self.file_path_dir_old[file_name_column].astype(str)
        self.file_path_dir_old = self.file_path_dir_old[~self.file_path_dir_old[file_name_column].str.contains('shin_sengoku')] #shin_sengoku removed from csv for file names for img paths (old collection)
        self.transform = transform

    def __len__(self):
        # correct csv file for new collection
        # '/content/drive/MyDrive/output_captioning/NFT_DATASET_MERGED/labels/labels_augmented.csv'
        # correct csv file for old collection 
        # '/content/drive/MyDrive/output_captioning/NFT_DATASET_MERGED/labels/labels_augmented_old_collection.csv'
        return len(self.labels_old) + len(self.labels_new)
    
    def __getitem__(self,index):
        try:
            if index >= len(self.labels_old):
                index = index - len(self.labels_old)
                img_dir = os.path.join(self.images_dir_new, self.labels_new.iloc[index].data_name)
                img = np.array(Image.open(img_dir).convert('RGB'))
                if self.transform:
                    img = self.transform(img)
                return img, self.labels_new.iloc['index'].__class__
            
            else:
                img_name = self.image_paths[index]
                img_path = os.path.join(self.image_dir, img_name)
                img_path = '/content/drive/MyDrive/output_captioning/NFT_DATASET_MERGED' + img_path
                img = np.array(Image.open(img_dir).convert('RGB'))
                if self.transform:
                    img = self.transform(img)
                return  img, self.labels_old.iloc[index].cls_label
            
        except Exception as e:
            raise RuntimeError(f"Error loading image at index {index}: {str(e)}")

def _arg_parse():
    args = argparse.ArgumentParser()
    args.add_argument("--images_dir_old", type=str)
    args.add_argument("--images_dir_new", type=str)
    args.add_argument("--split_ratio", type=float)
    args.add_argument("--label_dir_old", type=str)
    args.add_argument("--label_dir_new", type=str)
    args.add_argument("--file_path_dir_old", type=str)
    args.add_argument("--file_name_column", type=str)
    args.add_argument("--num_epochs", type=int)
    args.add_argument("--train", action="store_true")
    args.add_argument("--checkpoint", type=str)
    args.add_argument("--save_path", type=str)
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
    if args.train():
        dataset = RarityDataset(label_dir_old=args.label_dir_old,label_dir_new=args.label_dir_new,
                                image_dir_old=args.image_dir_old, image_dir_new=args.image_dir_new, transform=transform)
    else:
         dataset = RarityDataset(label_dir_old=args.label_dir_old, label_dir_new=args.label_dir_new,
                                      image_dir_old=args.images_dir_old, image_dir_new=args.images_dir_new, transform=test_transform)

    train_dataset, val_dataset = random_split(dataset, [int(len(dataset)*args.split_ratio), int(len(dataset) - int(len(dataset)*args.split_ratio))])
    print(len(train_dataset))
    print(len(val_dataset))
    if args.train:
        print("Training set size:", len(train_dataset))
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        print("Val set size:", len(val_dataset))
    else:
        print("Test set size:", len(val_dataset))
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224-in21k')
    model.classifier = nn.Linear(model.config.hidden_size, 1) 
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    if args.train:
        for epoch in range(args.num_epochs):
            train_loss = 0
            for i, batch in enumerate(train_loader):
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                labels = labels.unsqueeze(1).to(outputs.logits.dtype)

                probs = torch.sigmoid(outputs.logits)
                loss = criterion(probs, labels)
                loss.backward()
                optimizer.step()
                print(f"iteration {i}, loss: {loss.item()}")
                train_loss += loss.item()
            val_loss = val(model, val_loader, device, criterion=criterion)
            print(f"Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {train_loss / len(train_loader)} val_loss: {val_loss}")
        torch.save(model.state_dict(), args.save_path)
    else:
        if args.checkpoint:
            model.to('cuda')
            model.load_state_dict(torch.load(args.checkpoint))
            model.eval()

            true_labels = []
            predicted_labels = []
            print("========================================")
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    img, label = batch
                    output = model(img.to("cuda"))
                    true_labels.extend(label.numpy())
                    predicted_labels.extend((torch.sigmoid(output.logits).detach().cpu().numpy() > 0.5).astype(int))
            true_labels = np.array(true_labels)
            predicted_labels = np.array(predicted_labels)

            acc = accuracy_score(true_labels, predicted_labels)
            f1 = f1_score(true_labels, predicted_labels)
            p_score = precision_score(true_labels, predicted_labels)

            conf_matrix = confusion_matrix(true_labels, predicted_labels)

            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
            plt.show()
            plt.savefig("./conf.png")

            print(f'Accuracy: {acc}')
            print(f'F1 Score: {f1}')
            print(f'Precision Score: {p_score}')

if __name__ == "__main__":
    _train()