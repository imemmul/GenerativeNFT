import os
import pandas as pd
import numpy as np
import cv2
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

class_labels = pd.read_csv('/Users/beyzakaya/Desktop/bk/Akademik/Senior Design Project/rarity/old_dataset_rarity.csv')
classes = class_labels.copy()
old_data_dir = '/Users/beyzakaya/Desktop/bk/Akademik/Senior Design Project/rarity/nft_dataset_old/NFT_DATASET_MERGED/train'
augment_data_dir = '/Users/beyzakaya/Desktop/bk/Akademik/Senior Design Project/rarity/nft_dataset_old/NFT_DATASET_AUGMENTED'
#print(class_labels.columns)
#print(len(class_labels))

collection_names = []

for root,dirs, files in os.walk(old_data_dir):
    for directory in dirs:
        collection_names.append(directory)
collection_names = [name for name in collection_names if not name.startswith('.')]
#print(collection_names)

start_index = class_labels[class_labels['onChainName'] == 'Shadowy Super Coder #4383'].index[0]
end_index = class_labels[class_labels['onChainName'] == 'DeGod #839'].index[0]
start_index +=1
end_index -=1
#print("Start index:", start_index)
#print("End index:", end_index)
start_onChainName = class_labels.loc[start_index, 'onChainName']
end_onChainName = class_labels.loc[end_index, 'onChainName']
#print("Start onChainName:", start_onChainName)
#print("End onChainName:", end_onChainName)



# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.simplefilter(action='ignore', category=UserWarning)
# warnings.filterwarnings("ignore", message=".*loadsave.cpp.*")
for i in range(len(classes)):
    img_name = classes['onChainName'][i]
    
    # Remain
    if start_index <= i <= end_index:

        collection_name = 'shin_sengoku'
        if "'" in img_name:
            img_name = img_name.replace("'", "_")
        elif "ÅŒ" in img_name:
            img_name = img_name.replace("ÅŒ", "Ō")
        elif "Å«" in img_name and "ÅŌ" not in img_name:
            img_name = img_name.replace("Å«", "ū")
        
        special_cases = {
            "Genâ€™ichi Takemi": "Gen’ichi Takemi",
            "Genâ€™ichi Aragaki": "Gen’ichi Aragaki",
            "Ken_yÅ« Yanagimachi": "Ken_yū Yanagimachi",
            "Ken_yÅ« Uesaka": "Ken_yū Uesaka",
            "Ken_yÅ« Horihata": "Ken_yū Horihata",
            "Ken_yÅ« Mitsumori": "Ken_yū Mitsumori"
        }
        if img_name in special_cases:
            img_name = special_cases[img_name]
            img_dir = os.path.join(old_data_dir,collection_name,img_name)
            #print(img_name)
        
        img_name_with_extension = img_name + ".png"
        img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
        #print(f"Image dir shin_sengoku: {img_dir}")

        #print(img_dir)
        #img = cv2.imread(img_dir)
        #cv2.imshow(img)

        # special_cases = {
        #     "Kan'ichi Toriyabe": "Kan_ichi Toriyabe",
        #     "YÅ«ya Jinno": "Yūya Jinno",
        #     "RyÅ«suke Katsuno": "Ryūsuke Katsuno",
        #     "YÅ«dai Iwakabe": "Yūdai Iwakabe",
        #     "YÅ«suke Morinoue": "Yūsuke Morinoue",
        #     "Jun'ya Umemoto": "Jun_ya Umemoto",
        #     "Ken'ya Mitsukuri": "Ken_ya Mitsukuri",
        #     "JÅ«bei Manaka": "Jūbei Manaka",
        #     "RyÅ«shi Asaka": "Ryūshi Asaka",
        #     "ShÅ«suke Hara": "Shūsuke Hara",
        #     "YÅ«saku Sakemoto": "Yūsaku Sakemoto",
        #     "Ken'ichi Hiyama": "Ken_ichi Hiyama"

        # }
        if os.path.isfile(img_dir):
            pass
        else:
            print(f"not correct: {img_dir}")

    elif 'Degen Fat Cat' in img_name:
        collection_name = 'degenfatcats'
        img_number = img_name.split('#')[-1].strip()
        # print(f"Image number creation degenfatcat: {img_number}")
        img_number = ''.join(filter(str.isdigit, img_number))
        # print("Numeric part of the image name:", img_number)

        special_cases = {
            "17213": "Degen Fat Cat the 17213th.png",
            "18512": "Degen Fat Cat the 18512th.png",
            "19813": "Degen Fat Cat the 19813th.png",
            "15811": "Degen Fat Cat the 15811th.png",
            "12213": "Degen Fat Cat the 12213th.png",
            "13612": "Degen Fat Cat the 13612th.png",
            "6711": "Degen Fat Cat the 6711th.png"
        }

        if img_number in special_cases:
            img_name = special_cases[img_number]
            img_dir = os.path.join(old_data_dir,collection_name,img_name) 
        else:
            if 'Degen Fat Cat' in img_number:
                img_number = int(img_number)
                if img_number % 10 == 1 and img_number != 11:
                    suffix = "st"
                elif img_number % 10 == 2 and img_number != 12:
                    suffix = "nd"
                elif img_number % 10 == 3 and img_number != 13:
                    suffix = "rd"
                else:
                    suffix = "th"
                img_number_with_suffix = f"the {img_number}{suffix}"
                img_name_with_extension = img_number_with_suffix + ".png"
                img_dir = os.path.join(old_data_dir, collection_name, img_name_with_extension)

            else: # FIXME
                img_number = int(img_number)
                if img_number % 10 == 1 and img_number != 11:
                    suffix = "st"
                elif img_number % 10 == 2 and img_number != 12:
                    suffix = "nd"
                elif img_number % 10 == 3 and img_number != 13:
                    suffix = "rd"
                else:
                    suffix = "th"
                img_number_with_suffix = f"the {img_number}{suffix}"
                img_name_with_extension = f"Degen Fat Cat {img_number_with_suffix}.png"
                img_dir = os.path.join(old_data_dir, collection_name, img_name_with_extension)
        # print(img_number_with_suffix)
        img_name_with_extension = f"Degen Fat Cat {img_number_with_suffix}.png"
        img_name_with_extension = img_name + ".png"
        # print(img_name_with_extension)
        # print(img_dir)
        noluyo = []
        if os.path.isfile(img_dir):
            pass
        else:
            suffix = "th"
            img_number_with_suffix = f"the {img_number}{suffix}"
            #print(img_number_with_suffix)
            img_name_with_extension = f"Degen Fat Cat {img_number_with_suffix}.png"
            #print(img_name_with_extension)
            img_dir = os.path.join(old_data_dir, collection_name, img_name_with_extension)
            if os.path.isfile(img_dir):
                continue
                #print(f"fixed")
                
        if os.path.isfile(img_dir):
            pass
        else:
            print(f"not correct: {img_dir}")

    # Image names match with csv file names
    elif 'Degen Ape' in img_name:
        collection_name = 'degenerate_ape_academy'
        img_name_with_extension = img_name + ".png"
        img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
        #print(img_dir)
        if os.path.isfile(img_dir):
            pass
        else:
            print(f"not correct: {img_dir}")

    
    # Image names match with csv file names
    elif 'DeGod' in img_name:
        collection_name = 'Degods'
        img_name_with_extension = img_name + ".png"
        img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
        #print(img_dir)
        if os.path.isfile(img_dir):
            pass
        else:
            print(f"not correct: {img_dir}")
    
     # Image names match with csv file names
    elif 'SMB' in img_name:
        collection_name = 'solana_monkey_business'
        img_name_with_extension = img_name + ".png"
        img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
        #print(img_dir)
        if os.path.isfile(img_dir):
            pass
        else:
            print(f"not correct: {img_dir}")
    
    # Image names match with csv file names
    elif 'Fox' in img_name:
        collection_name = 'famous_fox_federation'
        img_name_with_extension = img_name + ".png"
        img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
        #print(img_dir)
        #img = cv2.imread(img_dir)
        if os.path.isfile(img_dir):
            pass
        else:
            print(f"not correct: {img_dir}")
    
    # Image names match with csv file names
    elif 'Okay Bear' in img_name:
        collection_name = 'Okay Bears'
        img_name_with_extension = img_name + ".png"
        img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
        #print(img_dir)
        #img = cv2.imread(img_dir)
        if os.path.isfile(img_dir):
            pass
        else:
            print(f"not correct: {img_dir}")
    
    # Image names match with csv file names
    elif 'Shadowy Super Coder' in img_name:
        collection_name = 'shadowy_super_coder_dao'
        img_name_with_extension = img_name + ".png"
        img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
        #print(img_dir)
        #img = cv2.imread(img_dir)
        if os.path.isfile(img_dir):
            pass
        else:
            print(f"not correct: {img_dir}")
    
    # Image names match with csv file names
    elif 'Remnants' in img_name:
        collection_name = 'the_remnants_'
        img_name_with_extension = img_name + ".png"
        img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
        #print(img_dir)
        if os.path.isfile(img_dir):
            pass
        else:
            print(f"not correct: {img_dir}")
    
    # Image names match with csv file names
    elif 'Cet' in img_name:
        collection_name = 'cets_on_creck'
        img_name_with_extension = img_name + ".png"
        img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
        #print(img_dir)
        if os.path.isfile(img_dir):
            pass
        else:
            print(f"not correct: {img_dir}")
            

    # Image names match with csv file names 
    elif 'y00t' in img_name:
        collection_name = 'y00ts'
        img_name_with_extension = img_name + ".png"
        img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
        #print(img_dir)
        if os.path.isfile(img_dir):
            pass
        else:
            print(f"not correct: {img_dir}")

    # Image names and csv file names are not same 
    elif 'Smyth' in img_name or 'Blocksmith' in img_name:
        collection_name = 'blocksmith_labs'
        if 'Blocksmith' in img_name:
            img_name = img_name.replace('Blocksmith Labs', 'Smyth')
        img_name_with_extension = img_name + ".png"
        img_dir = os.path.join(old_data_dir, collection_name, img_name_with_extension)
        if os.path.isfile(img_dir):
            pass
        else:
            print(f"not correct: {img_dir}")
        #print(f"Blocksmith: {img_dir}")
