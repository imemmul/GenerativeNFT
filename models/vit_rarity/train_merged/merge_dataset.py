import pandas as pd
import os
import shutil

images_dir = "/home/emir/Desktop/dev/datasets/nft_dataset/NFT_DATASET_MERGED/train"
augmented_old_collection = "/home/emir/Desktop/dev/datasets/nft_dataset/NFT_DATASET_MERGED/train"
old_collection = "/home/emir/Desktop/dev/datasets/nft_dataset/NFT_DATASET_MERGED/old_collection"
count = 0
images_to_move = [images_dir, augmented_old_collection]
for images_dir in images_to_move:
    for f in os.listdir(images_dir):
        folder_dir = os.path.join(images_dir, f)
        if f != "new_collection" and f != "old_collection" and not f.endswith(".csv") and f != ".DS_Store":
            for i in os.listdir(folder_dir):
                src = os.path.join(folder_dir, i)
                dst = os.path.join(old_collection, i)
                if os.path.isfile(src):
                    print(f"src: {src}")
                    print(f"dst: {dst}")
                    count += 1
                    print(count)
                    shutil.move(src, dst)
                else:
                    print("NOLUYO AMK")
                
        
 

# image_dir = "/home/emir/Desktop/dev/datasets/nft_dataset/NFT_DATASET_MERGED/train/old_collection"
# old_aug_all = pd.read_csv("/home/emir/Desktop/dev/datasets/nft_dataset/old_dataset_labels_augmented_cleared.csv")
# print(len(old_aug_all))
# print(len(os.listdir(image_dir)))
# count = 0
# for i in range(len(old_aug_all)):
#     if "augmented" in old_aug_all.iloc[i].onChainName:
#         print(os.path.join(image_dir, old_aug_all.iloc[i].onChainName))
#         if os.path.isfile(os.path.join(image_dir, old_aug_all.iloc[i].onChainName)):
#             print(f"this is file")
#         else:
#             count += 1
#             print(f"dropping {old_aug_all.iloc[i].onChainName}")
#             old_aug_all.drop(old_aug_all[old_aug_all.onChainName==old_aug_all.iloc[i].onChainName].index, inplace=True)
#     else:
#         img_dir = old_aug_all.iloc[i].onChainName + ".png"
#         print(os.path.join(image_dir, img_dir))
#         if os.path.isfile(os.path.join(image_dir, img_dir)):
#             print(f"this is file")
#         else:
#             count += 1
#             print(f"dropping: {img_dir[:-4]}")
#             old_aug_all.drop(old_aug_all[old_aug_all.onChainName==old_aug_all.iloc[i].onChainName].index, inplace=True)
# print(count)
