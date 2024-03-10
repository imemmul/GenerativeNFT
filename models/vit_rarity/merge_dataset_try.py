import pandas as pd
import os

original_df = pd.read_csv('/Users/beyzakaya/Desktop/bk/Akademik/Senior Design Project/rarity/old_dataset_rarity.csv')
original_df.drop(columns='Unnamed: 0.1', inplace=True)
original_df.drop(columns='Unnamed: 0', inplace=True)
original_df.drop_duplicates(subset='onChainName', inplace=True)
#columns1 = set(original_df.columns)
augmented_df = pd.read_csv('/Users/beyzakaya/Desktop/bk/Akademik/Senior Design Project/rarity/nft_dataset_old/NFT_DATASET_AUGMENTED/old_dataset_labels_augmented.csv')
augmented_df.drop(columns='Unnamed: 0', inplace=True)
augmented_df.drop(columns='Unnamed: 0.1', inplace=True)
augmented_df.drop(columns='Unnamed: 0.2', inplace=True)
augmented_df.drop_duplicates(subset='onChainName', inplace=True)
#columns2 = set(augmented_df.columns)
common_columns = original_df.columns.intersection(augmented_df.columns)
#df1_common = pd.read_csv('/Users/beyzakaya/Desktop/bk/Akademik/Senior Design Project/rarity/old_dataset_rarity.csv', usecols=common_columns)
#df2_common = pd.read_csv('/Users/beyzakaya/Desktop/bk/Akademik/Senior Design Project/rarity/nft_dataset_old/NFT_DATASET_AUGMENTED/old_dataset_labels_augmented.csv', usecols=common_columns)
original_df_unique = original_df.drop_duplicates(subset='onChainName')
augmented_df_unique = augmented_df.drop_duplicates(subset='onChainName')
merged_df_common_columns = pd.concat([original_df_unique, augmented_df_unique])
merged_df_common_columns.sort_values(by=['onChainName'], inplace=True)
merged_df_common_columns.drop_duplicates(subset='onChainName', inplace=True)
#print(merged_df_common_columns['cls_label'].head())

#print(f"Len of merged df: {len(merged_df_common_columns)}")
merged_df_common_columns.to_csv('merged_old_dataset.csv', index=False)
#print(merged_df_common_columns['rank_values_merarity'].head())

new_df = pd.read_csv('/Users/beyzakaya/Desktop/bk/Akademik/Senior Design Project/rarity/labels_augmented.csv')
#print(f"Len of new dataset : {len(new_df)}")
#print(f"Overall merged dataset should be in len: {len(new_df)+len(merged_df_common_columns)}")
new_df.drop(columns='Unnamed: 0.1', inplace=True)
new_df.drop(columns='Unnamed: 0', inplace=True)
new_df.rename(columns={'data_name': 'onChainName'}, inplace=True)
new_df.rename(columns={'cls': 'cls_label'}, inplace=True)
#print(new_df['rank_values'].head())
merged_final = pd.concat([merged_df_common_columns, new_df], ignore_index=True)
#print(f"Len of merged final df: {len(merged_final)}")
merged_final.to_csv('merged_final.csv', index=False)
merged_final.sort_values(by=['onChainName'], inplace=True)


old_data_dir = '/Users/beyzakaya/Desktop/bk/Akademik/Senior Design Project/rarity/nft_dataset_old/NFT_DATASET_MERGED/train'
old_data_merged_dir = '/Users/beyzakaya/Desktop/bk/Akademik/Senior Design Project/rarity/nft_dataset_old/NFT_DATASET_AUGMENTED/train'
for i in range(len(merged_final)):
    img_name = merged_final['onChainName'][i]
    #print(f"Image name inside merged df: {img_name}")

    if 'Blocksmith Labs' in img_name and 'augmented' not in img_name:
        collection_name = 'blocksmith_labs'
        img_name_folder = img_name = img_name.replace('Blocksmith Labs', 'Smyth')
        img_name_with_extension = img_name_folder + '.png'
        img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
        if not os.path.exists(img_dir):
            print(f"Image directory does not exist: {img_dir}")

    elif 'Blocksmith Labs' in img_name and 'augmented' in img_name:
        collection_name = 'blocksmith_labs'
        img_dir = os.path.join(old_data_merged_dir,collection_name,img_name)
        if not os.path.exists(img_dir):
            print(f"Image directory does not exist: {img_dir}")
    
    elif 'Cet' in img_name and 'augmented' not in img_name:
        collection_name = 'cets_on_creck'
        img_name_with_extension = img_name + '.png'
        img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
        if not os.path.exists(img_dir):
            print(f"Image directory does not exist: {img_dir}")
    
    elif 'Cet' in img_name and 'augmented' in img_name:
        collection_name = 'cets_on_creck'
        img_dir = os.path.join(old_data_merged_dir,collection_name,img_name)
        if not os.path.exists(img_dir):
            print(f"Image directory does not exist: {img_dir}")
    
    elif 'Degen Ape' in img_name and 'augmented' not in img_name:
        collection_name = 'degenerate_ape_academy'
        img_name_with_extension = img_name + '.png'
        img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
        if not os.path.exists(img_dir):
            print(f"Image directory does not exist: {img_dir}")
    
    elif 'Degen Ape' in img_name and 'augmented' in img_name:
        collection_name = 'degenerate_ape_academy'
        img_dir = os.path.join(old_data_merged_dir,collection_name,img_name)
        if not os.path.exists(img_dir):
            print(f"Image directory does not exist: {img_dir}")
    
    elif 'Degen Fat Cat' in img_name and 'augmented' not in img_name:
        collection_name = 'degenfatcats'
        img_number = img_name.split('#')[-1].strip()
        img_name_with_extension = img_name + '.png'
        img_number = ''.join(filter(str.isdigit, img_number))
        img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
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
                if not os.path.exists(img_dir):
                    suffix = "th"
                    img_number_with_suffix = f"the {img_number}{suffix}"
                    #print(img_number_with_suffix)
                    img_name_with_extension = f"Degen Fat Cat {img_number_with_suffix}.png"
                    #print(img_name_with_extension)
                    img_dir = os.path.join(old_data_dir, collection_name, img_name_with_extension)
                    if not os.path.exists(img_dir):
                        print(f"Image directory does not exist: {img_dir}")

    elif 'Degen Fat Cat' in img_name and 'augmented' in img_name:
        collection_name = 'degenfatcats'
        img_dir = os.path.join(old_data_merged_dir,collection_name,img_name)
        if not os.path.exists(img_dir):
            print(f"Image directory does not exist: {img_dir}")
    
    elif 'Degod' in img_name and 'augmented' not in img_name:
        collection_name = 'Degods'
        img_name_with_extension = img_name + '.png'
        img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
        if not os.path.exists(img_dir):
            print(f"Image directory does not exist: {img_dir}")
    
    elif 'Degod' in img_name and 'augmented' in img_name:
        collection_name = 'Degods'
        img_dir = os.path.join(old_data_merged_dir,collection_name,img_name)
        if not os.path.exists(img_dir):
            print(f"Image directory does not exist: {img_dir}")
    
    elif 'Fox' in img_name and 'augmented' not in img_name:
        collection_name = 'famous_fox_federation'
        img_name_with_extension = img_name + '.png'
        img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
        if not os.path.exists(img_dir):
            print(f"Image directory does not exist: {img_dir}")
    
    elif 'Fox' in img_name and 'augmented' in img_name:
        collection_name = 'famous_fox_federation'
        img_dir = os.path.join(old_data_merged_dir,collection_name,img_name)
        if not os.path.exists(img_dir):
            print(f"Image directory does not exist: {img_dir}")
    
    elif 'Okay Bear' in img_name and 'augmented' not in img_name:
        collection_name = 'Okay Bears'
        img_name_with_extension = img_name + '.png'
        img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
        if not os.path.exists(img_dir):
            print(f"Image directory does not exist: {img_dir}")
    
    elif 'Okay Bear' in img_name and 'augmented' in img_name:
        collection_name = 'Okay Bears'
        img_dir = os.path.join(old_data_merged_dir,collection_name,img_name)
        if not os.path.exists(img_dir):
            print(f"Image directory does not exist: {img_dir}")
    
    elif 'Shadowy Super Coder' in img_name and 'augmented' not in img_name:
        collection_name = 'shadowy_super_coder_dao'
        img_name_with_extension = img_name + '.png'
        img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
        if not os.path.exists(img_dir):
            print(f"Image directory does not exist: {img_dir}")
    
    elif 'Shadowy Super Coder' in img_name and 'augmented' in img_name:
        collection_name = 'shadowy_super_coder_dao'
        img_dir = os.path.join(old_data_merged_dir,collection_name,img_name)
        if not os.path.exists(img_dir):
            print(f"Image directory does not exist: {img_dir}")
    
    elif 'SMB' in img_name and 'augmented' not in img_name:
        collection_name = 'solana_monkey_business'
        img_name_with_extension = img_name + '.png'
        img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
        if not os.path.exists(img_dir):
            print(f"Image directory does not exist: {img_dir}")
    
    elif 'SMB' in img_name and 'augmented' in img_name:
        collection_name = 'solana_monkey_business'
        img_dir = os.path.join(old_data_merged_dir,collection_name,img_name)
        if not os.path.exists(img_dir):
            print(f"Image directory does not exist: {img_dir}")
        
    elif ('Female Remnants' or 'The Remnants' in img_name) and 'augmented' not in img_name:
        collection_name = 'the_remnants_'
        img_name_with_extension = img_name + '.png'
        img_dir = os.path.join(old_data_dir,collection_name,img_name_with_extension)
        if not os.path.exists(img_dir):
            print(f"Image directory does not exist: {img_dir}")



                


            
    








# duplicates = merged_df_common_columns[merged_df_common_columns.duplicated('onChainName', keep=False)]
# non_augmented_mask = ~merged_df_common_columns['onChainName'].str.contains('augmented', case=False)
# count_duplicate = 0
# for index, row in merged_df_common_columns.iterrows():
#     if non_augmented_mask.iloc[index]:
#         onChainName = row['onChainName']
#         if (merged_df_common_columns['onChainName'] == onChainName).sum() > 1:
#             count_duplicate += 1
# print("Count of duplicate non-augmented names:", count_duplicate)

#print("Duplicate onChainNames values:")
#print(duplicates['onChainName'].unique())
#print(f"{len(duplicates)} number of duplicates occur.")

#print(f"Original df's onChainName dtype: {original_df['onChainName'].dtype}")
#print(f"Augmented df's onChainName dtype: {augmented_df['onChainName'].dtype}")
#print(f"Original df's dtypes: {original_df.dtypes}")
#print(f"Augmented df's dtypes: {augmented_df.dtypes}")

