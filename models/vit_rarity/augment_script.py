import os
import pandas as pd
import numpy as np
from torchvision import transforms
import cv2
from imgaug import augmenters as iaa


transform = iaa.Sequential([
    iaa.Resize({"height": 512, "width": 512}),
    iaa.Fliplr(0.5),              # horizontal flips
    iaa.Crop(percent=(0, 0.1)),   # random crops
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),  # apply Gaussian blur with 50% probability
    iaa.Affine(rotate=(-45, 45)),  # random rotations
    iaa.AddToHueAndSaturation(value=(-20, 20)),  # change hue and saturation
    iaa.Multiply((0.5, 1.5), per_channel=0.5),  # random brightnessg
])


classes = pd.read_csv("./labels_classes.csv")
classes_copy = classes.copy()
data_dir = "/Users/emirulurak/Desktop/dev/ozu/openseadata/dataset/rarity_dataset"
augment_data_dir = "/Users/emirulurak/Desktop/dev/ozu/openseadata/dataset/rarity_dataset_augmented"
classes.drop(columns='Unnamed: 0', inplace=True)
print(classes.columns)
for i in range(len(classes)):
    img_dir = os.path.join(data_dir, classes['data_name'][i])
    cls = classes['cls'][i]
    if img_dir.endswith('png'):
        if cls:
            for _ in range(20):
                augmented_name = classes['data_name'][i][:-4] + f"_augmented_{_}.png"
                print(f"{augmented_name} : {cls}")
                folder_name = os.path.join(data_dir, augmented_name)
                new_row = {'data_name': augmented_name, 'label':classes['label'][i], 'dict':classes['dict'][i], 'rank_values':classes['rank_values'][i], 'cls':classes['cls'][i]}
                # classes_copy.loc[len(classes.index)] = [augmented_name, classes['label'][i], classes['dict'][i], classes['rank_values'][i], classes['cls'][i]]
                classes_copy.loc[len(classes_copy.index)] = new_row
                # img = cv2.imread(img_dir)
                # transformed = transform(images=[img])
                # cv2.imwrite(filename=os.path.join(augment_data_dir, augmented_name), img=transformed[0])
        else:
            pass
classes_copy.to_csv("./labels_augmented.csv")