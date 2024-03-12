import pandas as pd
import os

merged_df = pd.read_csv('/Users/beyzakaya/Desktop/bk/Akademik/Senior Design Project/rarity/GenerativeNFT/models/vit_rarity/merged_final.csv')

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

special_cases = {
    "Genâ€™ichi Takemi": "Gen’ichi Takemi",
    "Genâ€™ichi Aragaki": "Gen’ichi Aragaki",
    "Ken_yÅ« Yanagimachi": "Ken_yū Yanagimachi",
    "Ken_yÅū Uesaka": "Ken_yū Uesaka",
    "Ken_yÅ« Horihata": "Ken_yū Horihata",
    "Ken_yÅū Mitsumori": "Ken_yū Mitsumori"
}

old_data_dir = '/Users/beyzakaya/Desktop/bk/Akademik/Senior Design Project/rarity/nft_dataset_old/NFT_DATASET_MERGED/train'
old_data_merged_dir = '/Users/beyzakaya/Desktop/bk/Akademik/Senior Design Project/rarity/nft_dataset_old/NFT_DATASET_AUGMENTED/train'
new_data_dir = '/Users/beyzakaya/Desktop/bk/Akademik/Senior Design Project/rarity/nft_dataset_new/new_collection'

def get_image_dir(row):
    img_name = row['onChainName']
   
    if 'Degen Fat Cat' in img_name and 'augmented' not in img_name:
        collection_name = 'degenfatcats'
        img_number = img_name.split('#')[-1].strip()
        img_name_with_extension = img_name + '.png'
        img_number = ''.join(filter(str.isdigit, img_number))
        img_dir = os.path.join(old_data_dir, collection_name, img_name_with_extension)
        if not os.path.exists(img_dir):
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
                    img_name_with_extension = f"Degen Fat Cat {img_number_with_suffix}.png"
                    img_dir = os.path.join(old_data_dir, collection_name, img_name_with_extension)
                    if not os.path.exists(img_dir):
                        print(f"Image directory does not exist: {img_dir}")
    
    else:
        for keyword in collection_mappings:
            if keyword in img_name:
                collection_name = collection_mappings[keyword]
                if 'augmented' in img_name:
                    img_dir = os.path.join(old_data_merged_dir, collection_name, img_name)
                else:
                    img_name_folder = img_name.replace(keyword, 'Smyth')
                    img_dir = os.path.join(old_data_dir, collection_name, f'{img_name_folder}.png')
                return img_dir

        if img_name in special_cases:
            img_name = special_cases[img_name]

        if 'shin_sengoku' in str(row.values):
            collection_name = 'shin_sengoku'
            if 'augmented' in img_name:
                img_dir = os.path.join(old_data_dir, collection_name, img_name)
            else:
                img_dir = os.path.join(old_data_merged_dir, collection_name, f'{img_name}.png')
            return img_dir

        new_dataset_keywords = ['azragames', 'azuki', 'bastard-gan', 'beanzofficial', 'genesis-creepz', 'genuine-undead',
                                'kanpai-pandas', 'killabears', 'lazy-lions', 'ninja-squad-official', 'parallel-avatars',
                                'pixelmongen1', 'pudgypenguins', 'sappy-seals', 'thewarlords']
        if any(keyword in img_name for keyword in new_dataset_keywords):
            return os.path.join(new_data_dir, img_name)

    return "" 

merged_df['img_dir'] = merged_df.apply(get_image_dir, axis=1)
#print(merged_df['img_dir'].head())

collection_names = merged_df['img_dir'].apply(lambda x: x.split('/')[11] if len(x.split('/')) > 11 else "Other")

invalid_dirs_count_by_collection = (merged_df[~merged_df['img_dir'].apply(os.path.exists)]
                                   .groupby(collection_names)
                                   .size()
                                   .rename('Invalid Directories Count'))
print("Number of invalid image directories by collection:")
print(invalid_dirs_count_by_collection)


#invalid_dirs_count = merged_df['img_dir'].apply(lambda x: not os.path.exists(x) if x else False).sum()
#print(f"Number of invalid image directories: {invalid_dirs_count}")
