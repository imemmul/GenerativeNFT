import torch
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt


from torch.utils.data import DataLoader
from data.custom_dataset import CustomDataset  
from util.visualizer import Visualizer


train_dir = '/Users/beyzakaya/Desktop/Beyza Kaya /Akademik/Senior Design Project/GenerativeNFT/dataset/train'
test_dir = '/Users/beyzakaya/Desktop/Beyza Kaya /Akademik/Senior Design Project/GenerativeNFT/dataset/test'

# transform = transforms.Compose([
#     transforms.Resize((256,256)),
#     transforms.ToTensor()
# ])

# from torch.utils.data import DataLoader
# from torchvision.datasets import ImageFolder

# batch_size = 8
# train_dataset = ImageFolder(root=train_dir, transform=transform)
# test_dataset = ImageFolder(root=test_dir, transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count)

# for batch in train_loader:
#     images, labels = batch

#     for i in range(batch_size):
#         plt.imshow(images[i].permute(1,2,0))
#         plt.title(f'label: {labels[i]}')
#         plt.show()

import os
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = os.listdir(root_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data[idx])
        image = Image.open(img_name)
        print(f'number of color channels: {image.mode}')

        if self.transform:
            image = self.transform(image)
        print(f"img: {img_name} size: {image.shape}")
        return image

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

batch_size = 4  
train_dataset = CustomDataset(root_dir=train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

import matplotlib.pyplot as plt
import torchvision.utils as vutils

dl = iter(train_loader)
sample = next(dl)
print(sample)

# for idx, batch in enumerate(train_loader):
#     for i in range(batch.shape[0]):  # Iterate through images in the batch
        # image = batch[i]
        # mode = image.mode
        # print(f"Image {idx * batch_size + i + 1} - Mode: {mode}")

    # grid = vutils.make_grid(batch, normalize=True, scale_each=True)
    # plt.figure(figsize=(8, 8))
    # plt.imshow(grid.permute(1, 2, 0))
    # plt.axis('off')
    # plt.show()
    # break

for idx, batch in enumerate(train_loader):
    grid=vutils.make_grid(batch,normalize=True, scale_each=True)
    plt.figure(figsize=(8,8))
    plt.imshow(grid.permute(1,2,0))
    plt.axis(False)
    plt.show()


model = Pix2PixModel()
visualizer = Visualizer()
num_epochs = 50

for epoch in range(num_epochs):
    for idx, batch in enumerate(train_loader):
        model.set_input(batch)
        model.optimize_parameters()

    
    if epoch % 10 == 0:  
        visuals = model.get_current_visuals()
        visualizer.display_current_results(visuals, epoch, idx)

model.save_networks('trained_model')

for batch in train_loader:
    with torch.no_grad():
        model.set_input(batch)
        model.forward()
        output = model.get_current_visuals()  

    # Display the generated images
    grid = output['fake_B']  
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.show()
    break









