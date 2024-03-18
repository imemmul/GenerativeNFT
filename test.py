import os

dir = "/Users/emirulurak/Desktop/dev/ozu/openseadata/dataset/nft_dataset/old_dataset_labels_augmented.csv"

for i in os.listdir(dir):
    if os.path.isdir(os.path.join(dir, i)):
        print(i)
        print(len(os.listdir(os.path.join(dir, i))))