import pandas as pd
import os
import os
import pandas as pd
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa
from PIL import Image
import warnings


transform = iaa.Sequential([
    iaa.Resize({"height": 512, "width": 512}),
    iaa.Fliplr(0.5),              # horizontal flips
    iaa.Crop(percent=(0, 0.1)),   # random crops
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),  # apply Gaussian blur with 50% probability
    iaa.Affine(rotate=(-45, 45)),  # random rotations
    iaa.AddToHueAndSaturation(value=(-20, 20)),  # change hue and saturation
    iaa.Multiply((0.5, 1.5), per_channel=0.5),  # random brightnessg
])

dataset_dir = "/Users/emirulurak/Desktop/dev/ozu/openseadata/dataset/nft_dataset/NFT_DATASET_MERGED/train"
df = pd.read_csv("/Users/emirulurak/Desktop/dev/ozu/openseadata/dataset/new_old_collection_159_label.csv")
df_copy = df.copy()
for i in df[df['cls_label']==1].index:
    # print(f"{i}: {df.iloc[i]['data_name']}")
    exist = False
    for folder in os.listdir(dataset_dir):
        img_dir = os.path.join(dataset_dir, folder, df.iloc[i]['data_name'] + ".png")
        if os.path.isfile(img_dir):
            for _ in range(20):
                augmented_name = df.iloc[i]['data_name'] + f"_augmented_{_}"
                print(f"{augmented_name} : {df.iloc[i]['cls_label']}")
                folder_name = os.path.join(dataset_dir,folder)
                new_row = {'data_name': augmented_name, 'rarity':df.iloc[i]['rarity'], 'dict':df.iloc[i]['dict'], 'rank_values_moonrank':df.iloc[i]['rank_values_moonrank'], 'label':df.iloc[i]['label'], "cls_label": df.iloc[i]['cls_label']}
                max_index = df_copy.index.max()
                new_index = max_index + 1 if pd.notna(max_index) else 0

                df_copy.loc[new_index] = new_row
                img = np.array(Image.open(img_dir).convert('RGB'))
                transformed = transform.augment_image(img)
                Image.fromarray(transformed).save(os.path.join(os.path.join(folder_name, augmented_name+".png")))
        else:
            pass
df_copy.to_csv("./labels_augmented_old_collection.csv")

    

# class_labels = pd.read_csv('/Users/beyzakaya/Desktop/bk/Akademik/Senior Design Project/rarity/old_dataset_rarity.csv')
# classes_copy = class_labels.copy()
# old_data_dir = '/Users/beyzakaya/Desktop/bk/Akademik/Senior Design Project/rarity/nft_dataset_old/NFT_DATASET_MERGED/train'
# augment_data_dir = '/Users/beyzakaya/Desktop/bk/Akademik/Senior Design Project/rarity/nft_dataset_old/NFT_DATASET_AUGMENTED/train'
# #print(class_labels.columns)
# #print(len(class_labels))
# #print(f"Columns of csv: {classes.columns}")
# class_labels.drop(columns='Unnamed: 0.1', inplace=True)
# #print(f"Classes of csv after drop 1: {class_labels.columns}")
# class_labels.drop(columns='Unnamed: 0', inplace=True)
# #print(f"Classes of csv after drop 2: {class_labels.columns}")

# collection_names = []

# for root,dirs, files in os.walk(old_data_dir):
#     for directory in dirs:
#         collection_names.append(directory)
# collection_names = [name for name in collection_names if not name.startswith('.')]
# #print(collection_names)

# start_index = class_labels[class_labels['onChainName'] == 'Shadowy Super Coder #4383'].index[0]
# end_index = class_labels[class_labels['onChainName'] == 'DeGod #839'].index[0]
# start_index +=1
# end_index -=1
# #print("Start index:", start_index)
# #print("End index:", end_index)
# start_onChainName = class_labels.loc[start_index, 'onChainName']
# end_onChainName = class_labels.loc[end_index, 'onChainName']
# #print("Start onChainName:", start_onChainName)
# #print("End onChainName:", end_onChainName)



# # warnings.filterwarnings("ignore", category=UserWarning)
# # warnings.simplefilter(action='ignore', category=UserWarning)
# # warnings.filterwarnings("ignore", message=".*loadsave.cpp.*")
# for i in range(len(class_labels)):
#     img_name = class_labels['onChainName'][i]
    
#     # Remain
#     if start_index <= i <= end_index:

#         collection_name = 'shin_sengoku'
#         if "'" in img_name:
#             img_name = img_name.replace("'", "_")
#         elif "ÅŒ" in img_name:
#             img_name = img_name.replace("ÅŒ", "Ō")
#         elif "Å«" in img_name and "ÅŌ" not in img_name:
#             img_name = img_name.replace("Å«", "ū")
        
#         special_cases = {
#             "Genâ€™ichi Takemi": "Gen’ichi Takemi",
#             "Genâ€™ichi Aragaki": "Gen’ichi Aragaki",
#             "Ken_yÅ« Yanagimachi": "Ken_yū Yanagimachi",
#             "Ken_yÅ« Uesaka": "Ken_yū Uesaka",
#             "Ken_yÅ« Horihata": "Ken_yū Horihata",
#             "Ken_yÅ« Mitsumori": "Ken_yū Mitsumori"
#         }
#         if img_name in special_cases:
#             img_name = special_cases[img_name]
#             img_dir = os.path.join(old_data_dir,collection_name,img_name)
#             #print(img_name)
        
#         img_name_with_extension = img_name + ".png"
#         img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
#         #print(f"Image dir shin_sengoku: {img_dir}")

#         #print(img_dir)
#         #img = cv2.imread(img_dir)
#         #cv2.imshow(img)

#         # special_cases = {
#         #     "Kan'ichi Toriyabe": "Kan_ichi Toriyabe",
#         #     "YÅ«ya Jinno": "Yūya Jinno",
#         #     "RyÅ«suke Katsuno": "Ryūsuke Katsuno",
#         #     "YÅ«dai Iwakabe": "Yūdai Iwakabe",
#         #     "YÅ«suke Morinoue": "Yūsuke Morinoue",
#         #     "Jun'ya Umemoto": "Jun_ya Umemoto",
#         #     "Ken'ya Mitsukuri": "Ken_ya Mitsukuri",
#         #     "JÅ«bei Manaka": "Jūbei Manaka",
#         #     "RyÅ«shi Asaka": "Ryūshi Asaka",
#         #     "ShÅ«suke Hara": "Shūsuke Hara",
#         #     "YÅ«saku Sakemoto": "Yūsaku Sakemoto",
#         #     "Ken'ichi Hiyama": "Ken_ichi Hiyama"

#         # }
#         if os.path.isfile(img_dir):
#             pass
#         else:
#             print(f"not correct: {img_dir}")

#     elif 'Degen Fat Cat' in img_name:
#         collection_name = 'degenfatcats'
#         img_number = img_name.split('#')[-1].strip()
#         # print(f"Image number creation degenfatcat: {img_number}")
#         img_number = ''.join(filter(str.isdigit, img_number))
#         # print("Numeric part of the image name:", img_number)

#         special_cases = {
#             "17213": "Degen Fat Cat the 17213th.png",
#             "18512": "Degen Fat Cat the 18512th.png",
#             "19813": "Degen Fat Cat the 19813th.png",
#             "15811": "Degen Fat Cat the 15811th.png",
#             "12213": "Degen Fat Cat the 12213th.png",
#             "13612": "Degen Fat Cat the 13612th.png",
#             "6711": "Degen Fat Cat the 6711th.png"
#         }

#         if img_number in special_cases:
#             img_name = special_cases[img_number]
#             img_dir = os.path.join(old_data_dir,collection_name,img_name) 
#         else:
#             if 'Degen Fat Cat' in img_number:
#                 img_number = int(img_number)
#                 if img_number % 10 == 1 and img_number != 11:
#                     suffix = "st"
#                 elif img_number % 10 == 2 and img_number != 12:
#                     suffix = "nd"
#                 elif img_number % 10 == 3 and img_number != 13:
#                     suffix = "rd"
#                 else:
#                     suffix = "th"
#                 img_number_with_suffix = f"the {img_number}{suffix}"
#                 img_name_with_extension = img_number_with_suffix + ".png"
#                 img_dir = os.path.join(old_data_dir, collection_name, img_name_with_extension)

#             else: # FIXME
#                 img_number = int(img_number)
#                 if img_number % 10 == 1 and img_number != 11:
#                     suffix = "st"
#                 elif img_number % 10 == 2 and img_number != 12:
#                     suffix = "nd"
#                 elif img_number % 10 == 3 and img_number != 13:
#                     suffix = "rd"
#                 else:
#                     suffix = "th"
#                 img_number_with_suffix = f"the {img_number}{suffix}"
#                 img_name_with_extension = f"Degen Fat Cat {img_number_with_suffix}.png"
#                 img_dir = os.path.join(old_data_dir, collection_name, img_name_with_extension)
#         # print(img_number_with_suffix)
#         img_name_with_extension = f"Degen Fat Cat {img_number_with_suffix}.png"
#         img_name_with_extension = img_name + ".png"
#         # print(img_name_with_extension)
#         # print(img_dir)
#         noluyo = []
#         if os.path.isfile(img_dir):
#             pass
#         else:
#             suffix = "th"
#             img_number_with_suffix = f"the {img_number}{suffix}"
#             #print(img_number_with_suffix)
#             img_name_with_extension = f"Degen Fat Cat {img_number_with_suffix}.png"
#             #print(img_name_with_extension)
#             img_dir = os.path.join(old_data_dir, collection_name, img_name_with_extension)
#             if os.path.isfile(img_dir):
#                 continue
#                 #print(f"fixed")
                
#         if os.path.isfile(img_dir):
#             pass
#         else:
#             print(f"not correct: {img_dir}")

#     # Image names match with csv file names
#     elif 'Degen Ape' in img_name:
#         collection_name = 'degenerate_ape_academy'
#         img_name_with_extension = img_name + ".png"
#         img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
#         #print(img_dir)
#         if os.path.isfile(img_dir):
#             pass
#         else:
#             print(f"not correct: {img_dir}")

    
#     # Image names match with csv file names
#     elif 'DeGod' in img_name:
#         collection_name = 'Degods'
#         img_name_with_extension = img_name + ".png"
#         img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
#         #print(img_dir)
#         if os.path.isfile(img_dir):
#             pass
#         else:
#             print(f"not correct: {img_dir}")
    
#      # Image names match with csv file names
#     elif 'SMB' in img_name:
#         collection_name = 'solana_monkey_business'
#         img_name_with_extension = img_name + ".png"
#         img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
#         #print(img_dir)
#         if os.path.isfile(img_dir):
#             pass
#         else:
#             print(f"not correct: {img_dir}")
    
#     # Image names match with csv file names
#     elif 'Fox' in img_name:
#         collection_name = 'famous_fox_federation'
#         img_name_with_extension = img_name + ".png"
#         img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
#         #print(img_dir)
#         #img = cv2.imread(img_dir)
#         if os.path.isfile(img_dir):
#             pass
#         else:
#             print(f"not correct: {img_dir}")
    
#     # Image names match with csv file names
#     elif 'Okay Bear' in img_name:
#         collection_name = 'Okay Bears'
#         img_name_with_extension = img_name + ".png"
#         img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
#         #print(img_dir)
#         #img = cv2.imread(img_dir)
#         if os.path.isfile(img_dir):
#             pass
#         else:
#             print(f"not correct: {img_dir}")
    
#     # Image names match with csv file names
#     elif 'Shadowy Super Coder' in img_name:
#         collection_name = 'shadowy_super_coder_dao'
#         img_name_with_extension = img_name + ".png"
#         img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
#         #print(img_dir)
#         #img = cv2.imread(img_dir)
#         if os.path.isfile(img_dir):
#             pass
#         else:
#             print(f"not correct: {img_dir}")
    
#     # Image names match with csv file names
#     elif 'Remnants' in img_name:
#         collection_name = 'the_remnants_'
#         img_name_with_extension = img_name + ".png"
#         img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
#         #print(img_dir)
#         if os.path.isfile(img_dir):
#             pass
#         else:
#             print(f"not correct: {img_dir}")
    
#     # Image names match with csv file names
#     elif 'Cet' in img_name:
#         collection_name = 'cets_on_creck'
#         img_name_with_extension = img_name + ".png"
#         img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
#         #print(img_dir)
#         if os.path.isfile(img_dir):
#             pass
#         else:
#             print(f"not correct: {img_dir}")
            

#     # Image names match with csv file names 
#     elif 'y00t' in img_name:
#         collection_name = 'y00ts'
#         img_name_with_extension = img_name + ".png"
#         img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
#         #print(img_dir)
#         if os.path.isfile(img_dir):
#             pass
#         else:
#             print(f"not correct: {img_dir}")

#     # Image names and csv file names are not same 
#     elif 'Smyth' in img_name or 'Blocksmith' in img_name:
#         collection_name = 'blocksmith_labs'
#         if 'Blocksmith' in img_name:
#             img_name = img_name.replace('Blocksmith Labs', 'Smyth')
#         img_name_with_extension = img_name + ".png"
#         img_dir = os.path.join(old_data_dir, collection_name, img_name_with_extension)
#         if os.path.isfile(img_dir):
#             pass
#         else:
#             print(f"not correct: {img_dir}")
#         #print(f"Blocksmith: {img_dir}")
    
#     cls = class_labels['cls_label'][i]
#     #print(f"Classes for index{i}: {cls}")
#     img_dir_count = 0
#     if img_dir.endswith('png'):
#         img_dir_count += 1
#         if cls == 1:
#             for _ in range(20): # 20 new augmentation for one rare image 20*150 = 3000 normalization for collection
#                 augmented_name = class_labels['onChainName'][i] + f"_augmented_{_}.png"
#                 #print(f"augmented name: {augmented_name}")
#                 augmented_folder_name = os.path.join(old_data_dir,augmented_name)
#                 #print(f"Augmented folder name: {augmented_folder_name}")

#                 #print(f"Label: {class_labels['label'][i]}")
#                 #print(f"Dict: {class_labels['dict'][i]} ")
#                 #print(f"Rank values: {class_labels['rank_values_merarity'][i]}")
#                 #print(f"Class Labels: {class_labels['cls_label'][i]}")

#                 new_row = {'onChainName': augmented_name, 'label':class_labels['label'][i], 'dict':class_labels['dict'][i], 'rank_values':class_labels['rank_values_merarity'][i], 'cls_label':class_labels['cls_label'][i]}
#                 #print(f"New row for csv file with index {i}: {new_row}")

#                 # Adding a new created row to already existing dataframe 
#                 classes_copy.loc[len(classes_copy.index)] = new_row
#                 classes_copy.drop(columns=['Unnamed: 0'])
#                 classes_copy.drop(columns=['Unnamed: 0.1'])
#                 #print(img_dir)
                
#                 try:
#                     #img = cv2.imread(img_dir)
#                     img = Image.open(img_dir)
#                     img = img.convert('RGB')
#                     transformed = transform.augment_image(np.array(img))
#                     print(f"transformed: {img_dir}")
#                     #transformed = transform(images=[img])

#                     #for transformed_img in transformed:
#                         #Image.fromarray(transformed_img).show()
                    
#                     #for i, image in enumerate(transformed):
#                     #    cv2.imshow(f"Augmented Image {i+1}", image)
#                     #    cv2.waitKey(1000)
#                     #    cv2.destroyWindow(f"Augmented Image {i+1}")

#                     #cv2.imwrite(filename=os.path.join(augment_data_dir, augmented_name), img=np.array(transformed))

#                     os.makedirs(os.path.join(augment_data_dir, collection_name), exist_ok=True)
#                     Image.fromarray(transformed).save(os.path.join(augment_data_dir, collection_name, augmented_name))
                
#                 except AssertionError as e:
#                     print(f"Error in image: {img_dir}")
#                     print(e)
        
#         else:
#             print(f"not rare, :{img_dir.split('/')[-1]}")

# classes_copy.to_csv("./old_dataset_labels_augmented.csv")  