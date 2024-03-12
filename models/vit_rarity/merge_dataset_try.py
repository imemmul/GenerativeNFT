import pandas as pd
import os

# original_df = pd.read_csv('/Users/beyzakaya/Desktop/bk/Akademik/Senior Design Project/rarity/old_dataset_rarity.csv')
# original_df.drop(columns='Unnamed: 0.1', inplace=True)
# original_df.drop(columns='Unnamed: 0', inplace=True)
# original_df.drop_duplicates(subset='onChainName', inplace=True)
# #columns1 = set(original_df.columns)
# augmented_df = pd.read_csv('/Users/beyzakaya/Desktop/bk/Akademik/Senior Design Project/rarity/nft_dataset_old/NFT_DATASET_AUGMENTED/old_dataset_labels_augmented.csv')
# augmented_df.drop(columns='Unnamed: 0', inplace=True)
# augmented_df.drop(columns='Unnamed: 0.1', inplace=True)
# augmented_df.drop(columns='Unnamed: 0.2', inplace=True)
# augmented_df.drop_duplicates(subset='onChainName', inplace=True)
# #columns2 = set(augmented_df.columns)
# common_columns = original_df.columns.intersection(augmented_df.columns)
# #df1_common = pd.read_csv('/Users/beyzakaya/Desktop/bk/Akademik/Senior Design Project/rarity/old_dataset_rarity.csv', usecols=common_columns)
# #df2_common = pd.read_csv('/Users/beyzakaya/Desktop/bk/Akademik/Senior Design Project/rarity/nft_dataset_old/NFT_DATASET_AUGMENTED/old_dataset_labels_augmented.csv', usecols=common_columns)
# original_df_unique = original_df.drop_duplicates(subset='onChainName')
# augmented_df_unique = augmented_df.drop_duplicates(subset='onChainName')
# merged_df_common_columns = pd.concat([original_df_unique, augmented_df_unique])
# merged_df_common_columns.sort_values(by=['onChainName'], inplace=True)
# merged_df_common_columns.drop_duplicates(subset='onChainName', inplace=True)
# #print(merged_df_common_columns['cls_label'].head())

# #print(f"Len of merged df: {len(merged_df_common_columns)}")
# merged_df_common_columns.to_csv('merged_old_dataset.csv', index=False)
# #print(merged_df_common_columns['rank_values_merarity'].head())

# new_df = pd.read_csv('/Users/beyzakaya/Desktop/bk/Akademik/Senior Design Project/rarity/labels_augmented.csv')
# #print(f"Len of new dataset : {len(new_df)}")
# #print(f"Overall merged dataset should be in len: {len(new_df)+len(merged_df_common_columns)}")
# new_df.drop(columns='Unnamed: 0.1', inplace=True)
# new_df.drop(columns='Unnamed: 0', inplace=True)
# new_df.rename(columns={'data_name': 'onChainName'}, inplace=True)
# new_df.rename(columns={'cls': 'cls_label'}, inplace=True)
# #print(new_df['rank_values'].head())
# merged_final = pd.concat([merged_df_common_columns, new_df], ignore_index=True)
# #print(f"Len of merged final df: {len(merged_final)}")
# merged_final.to_csv('merged_final.csv', index=False)
# merged_final.sort_values(by=['onChainName'], inplace=True)

merged_final = pd.read_csv('/Users/beyzakaya/Desktop/bk/Akademik/Senior Design Project/rarity/GenerativeNFT/models/vit_rarity/merged_final.csv')

old_data_dir = '/Users/beyzakaya/Desktop/bk/Akademik/Senior Design Project/rarity/nft_dataset_old/NFT_DATASET_MERGED/train'
old_data_merged_dir = '/Users/beyzakaya/Desktop/bk/Akademik/Senior Design Project/rarity/nft_dataset_old/NFT_DATASET_AUGMENTED/train'
new_data_dir = '/Users/beyzakaya/Desktop/bk/Akademik/Senior Design Project/rarity/nft_dataset_new/new_collection'

collection_mappings = {
    'Blocksmith Labs': 'blocksmith_labs',
    'Cet': 'cets_on_creck',
    'Degen Ape': 'degenerate_ape_academy',
    'Degen Fat Cat': 'degenfatcats',
    'Degod': 'Degods',
    'Fox': 'famous_fox_federation',
    'Okay Bear': 'Okay Bears',
    'Shadowy Super Coder': 'shadowy_super_coder_dao',
    'SMB': 'solana_monkey_business',
    'Female Remnants': 'the_remnants_',
    'The Remnants': 'the_remnants_',
    'y00t': 'y00ts'
}

#collection_names = set(collection_mappings.values())
#print(collection_names)

img_dirs = []

for index,row in merged_final.iterrows():
    img_name = merged_final['onChainName'][index]
    #print(f"Image name inside merged df: {img_name}")
    """
       Blocksmith Labs
       cets_on_creck
       degenerate_ape_academy
       famous_fox_federation
       Okay Bears
       solana_monkey_business
       the_remnants_
    collections done in here

       """
    if 'Blocksmith Labs' in img_name or 'Cet' in img_name or 'Degen Ape' in img_name or 'Fox' in img_name or 'Okay Bear' in img_name or 'SMB' in img_name or 'Remnants' in img_name:
        for keyword, collection_name in collection_mappings.items():
            if keyword in img_name:
                collection_name = collection_mappings[keyword]
                #print(f"Collection name: {collection_name}")
                break
        if 'augmented' not in img_name:
            img_name_folder = img_name = img_name.replace('Blocksmith Labs', 'Smyth')
            img_name_with_extension = img_name_folder + '.png'
            img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
            img_dirs.append(img_dir)
            if not os.path.exists(img_dir):
                print(f"Image directory does not exist: {img_dir}")
        else:
            img_dir = os.path.join(old_data_merged_dir,collection_name,img_name)
            img_dirs.append(img_dir)
            if not os.path.exists(img_dir):
                print(f"Image directory does not exist: {img_dir}")

    # y00ts and Degods collections done in here
    elif 'y00t' in img_name or 'Degod' in img_name:
        collection_name = collection_mappings.get('y00t')
        if 'augmented' not in img_name:
            img_name_folder = img_name = img_name.replace('Blocksmith Labs', 'Smyth')
            img_name_with_extension = img_name_folder + '.png'
            img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
            img_dirs.append(img_dir)
            if not os.path.exists(img_dir):
                print(f"Image directory does not exist: {img_dir}")
        else:
            img_dir = os.path.join(old_data_merged_dir,collection_name,img_name)
            img_dirs.append(img_dir)
            if not os.path.exists(img_dir):
                print(f"Image directory does not exist: {img_dir}")
    
    # Degen Fat Cat handled in here
    elif 'Degen Fat Cat' in img_name:
        collection_name = 'degenfatcats'
        img_number = img_name.split('#')[-1].strip()
        img_name_with_extension = img_name + '.png'
        img_number = ''.join(filter(str.isdigit, img_number))
        img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
        img_dirs.append(img_dir)
        if not os.path.exists(img_dir):
            #print(f"Image directory does not exist: {img_dir}")
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
                img_dirs.append(img_dir)
                if not os.path.exists(img_dir):
                    print(f"Image directory does not exist: {img_dir}")
            
            else: 
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
                img_dirs.append(img_dir)
                if not os.path.exists(img_dir):
                    suffix = "th"
                    img_number_with_suffix = f"the {img_number}{suffix}"
                    #print(img_number_with_suffix)
                    img_name_with_extension = f"Degen Fat Cat {img_number_with_suffix}.png"
                    #print(img_name_with_extension)
                    img_dir = os.path.join(old_data_dir, collection_name, img_name_with_extension)
                    img_dirs.append(img_dir)
                    #if not os.path.exists(img_dir):
                        #print(f"Image directory does not exist: {img_dir}")
    
    #elif ('azragames' or 'azuki' or 'bastard-gan' or 'beanzofficial' or 'genesis-creepz' or 'genuine-undead' or 'kanpai-pandas' or 'killabears' or 'lazy-lions' or 'ninja-squad-official' or 'parallel-avatars' or 'pixelmongen1' or 'pudgypenguins' or 'sappy-seals' or 'thewarlords') in img_name:
    elif any(keyword in img_name for keyword in ['azragames','azuki_','bastard-gan', 'beanzofficial', 'genesis-creepz', 'genuine-undead', 'kanpai-pandas', 'killabears', 'lazy-lions', 'ninja-squad-official', 'parallel-avatars', 'pixelmongen1', 'pudgypenguins', 'sappy-seals', 'thewarlords']):
        img_dir = os.path.join(new_data_dir,img_name)
        img_dirs.append(img_dir)
        if not os.path.exists(img_dir):
            print(f"Image directory does not exist: {img_dir}")

    elif 'shin_sengoku' in str(row.values):
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

        if 'augmented' not in img_name:
            img_name_with_extension = img_name + '.png'
            img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
            img_dirs.append(img_dir)
            if not os.path.exists(img_dir):
                if img_name in special_cases:
                    img_name = special_cases[img_name]
                    img_name_with_extension = img_name + '.png'
                    img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
                    img_dirs.append(img_dir)
                    if not os.path.exists(img_dir):
                        print(f"Image directory does not exist: {img_dir}")

        else:
            img_dir = os.path.join(old_data_dir,collection_name,img_name)
            img_dirs.append(img_dir)
            if not os.path.exists(img_dir):
                if img_name in special_cases:
                    img_name = special_cases[img_name]
                    img_dir =  os.path.join(old_data_dir,collection_name,img_name)
                    img_dirs.append(img_dir)
                    if not os.path.exists(img_dir):
                        print(f"Image directory does not exist: {img_dir}")

merged_final['img_dir'] = img_dir
print(f"Len of df: {len(merged_final)}")
print(f"Len of img_dirs: {len(img_dirs)}")

if len(merged_final) == len(img_dirs):
    print("Number of img_dir values matches the total number of lines in the DataFrame.")
    for i in range(5):
        print(f"First 5 image dirs : {img_dir[i]}")
else:
   missing_indices = {}
   for i, (index, row) in enumerate(merged_final.iterrows()):
        if i >= len(img_dirs):
            print(f"Index {i} exceeds the length of img_dirs.")
            break

        img_dir_parts = img_dirs[i].split('/')
        #print(img_dir_parts)
        if 'new_collection' not in img_dir_parts:
            collection_name = img_dir_parts[-2]
        else:
            pass
        print(collection_name)
        if collection_name not in missing_indices:
            missing_indices[collection_name] = []
        missing_indices[collection_name].append(index)

   for collection_name, indices in missing_indices.items():
       print(f"{collection_name}: {len(indices)} missing indices")
          
    

    # shin sengoku collection
    # else:
    #     #print(f"else block")
    #     for index, row in merged_final.iterrows():
    #         #print(f"shin sengoku")
    #         img_name = merged_final['onChainName'][index]
    #         if 'shin_sengoku' in str(row.values) and 'augmented' not in img_name:
    #             #print(f"Image name for shin_sengoku: {img_name}")
    #             collection_name = 'shin_sengoku'
    #             special_cases = {
    #                 "Genâ€™ichi Takemi": "Gen’ichi Takemi",
    #                 "Genâ€™ichi Aragaki": "Gen’ichi Aragaki",
    #                 "Ken_yÅ« Yanagimachi": "Ken_yū Yanagimachi",
    #                 "Ken_yÅ« Uesaka": "Ken_yū Uesaka",
    #                 "Ken_yÅ« Horihata": "Ken_yū Horihata",
    #                 "Ken_yÅ« Mitsumori": "Ken_yū Mitsumori"
    #             }
    #             if img_name in special_cases:
    #                 img_name = special_cases[img_name]
    #                 img_name_with_extension = img_name + '.png'
    #                 img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
    #                 if not os.path.exists(img_dir):
    #                     # if "'" in img_name:
    #                     #     img_name = img_name.replace("'", "_")
    #                     # elif "ÅŒ" in img_name:
    #                     #     img_name = img_name.replace("ÅŒ", "Ō")
    #                     # elif "Å«" in img_name and "ÅŌ" not in img_name:
    #                     #     img_name = img_name.replace("Å«", "ū")
    #                     # img_name_with_extension = img_name + ".png"
    #                     # img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
                        
    #                     #if not os.path.exists(img_dir):
    #                     print(f"Image directory does not exist: {img_dir}")
                
    #         elif 'shin_sengoku' in str(row.values) and 'augmented' in img_name: 
    #             collection_name = 'shin_sengoku'
    #             special_cases = {
    #                 "Genâ€™ichi Takemi": "Gen’ichi Takemi",
    #                 "Genâ€™ichi Aragaki": "Gen’ichi Aragaki",
    #                 "Ken_yÅ« Yanagimachi": "Ken_yū Yanagimachi",
    #                 "Ken_yÅ« Uesaka": "Ken_yū Uesaka",
    #                 "Ken_yÅ« Horihata": "Ken_yū Horihata",
    #                 "Ken_yÅ« Mitsumori": "Ken_yū Mitsumori"
    #             }
    #             if img_name in special_cases:
    #                 img_name = special_cases[img_name]
    #                 img_name_with_extension = img_name + '.png'
    #                 img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
    #                 if not os.path.exists(img_dir):
    #                     # if "'" in img_name:
    #                     #     img_name = img_name.replace("'", "_")
    #                     # elif "ÅŒ" in img_name:
    #                     #     img_name = img_name.replace("ÅŒ", "Ō")
    #                     # elif "Å«" in img_name and "ÅŌ" not in img_name:
    #                     #     img_name = img_name.replace("Å«", "ū")
    #                     # img_dir = os.path.join(old_data_dir,collection_name,img_name)
    #                     #if not os.path.exists(img_dir):
    #                     print(f"Image directory does not exist: {img_dir}")
                    

