import os
from download_dataset import load_json
import argparse
import pandas as pd

def check_null(data:dict):
    """
    loops all types and values and discards none values
    """
    if data["data"]:
        cleaned_data = [item for item in data["data"] if item["value"] != "None" and item["trait_type"] != "None"]
        if data["data"] != cleaned_data:
            pass
        return cleaned_data
    else:
        print(f"empty data")

def handler(nft_class:str, data:dict, img_dir:str, args):
    data["data"] = check_null(data=data)
    txt = ""
    if nft_class == "degenerate_ape_academy":
        for element in data["data"][::-1]:
            if len(element['trait_type'].split('/')) > 1:
                if len(element['value'].split('/')) > 1:
                    for i in range(len(element['trait_type'].split('/'))):
                        txt += element['value'].split('/')[i].strip() + " " + element['trait_type'].split('/')[i].strip() + ", "
                elif len(element['value'].split('/')) == 1:
                    for i in range(len(element['trait_type'].split('/'))):
                        txt += element['value'].split('/')[0].strip() + " " + element['trait_type'].split('/')[i].strip() + ", "
            elif element['trait_type'] != "sequence" and element['trait_type'] != "generation":
                txt += element['value'] + " " + element['trait_type'] + ", "
    elif nft_class == "degenfatcats":
         for element in data["data"][::-1]:
            if len(element['value'].split(',')) > 1:
                for i in range(len(element['value'].split(','))):
                    if i % 2 == 1:
                        txt += element['value'].split(',')[i].strip() + " "
                    else:
                        txt += element['value'].split(',')[i].strip() + " and "
                txt += element['trait_type'] + ", "
            else:
                txt += element['value'] + " " + element['trait_type'] + ", "
    elif nft_class == "Degods":
        for element in data["data"][::-1]:
            if element['trait_type'] != "y00t":
                txt += element['value'] + " " + element['trait_type'] + ", "
    elif nft_class == "famous_fox_federation":
        for element in data["data"][::-1]:
            if "Tier" in element['trait_type']:
                txt += element['trait_type'] + " " + str(element['value']) + ", "
            else:
                txt += element['value'] + " " + element['trait_type'] + ", "
    elif nft_class == "solana_monkey_business":
        for element in data["data"][::-1]:
            if element["trait_type"] != "Attributes Count":
                txt += element['value'] + " " + element['trait_type'] + ", "
    elif nft_class == "y00ts":
        for element in data["data"][::-1]:
            if element["trait_type"] != "1/1":
                txt += element['value'] + " " + element['trait_type'] + ", "
    else:
        for element in data["data"][::-1]:
            txt += element['value'] + " " + element['trait_type'] + ", "
    if args.verbose:
        print(f"{nft_class}: {txt[:-2]}\n")
    txt = txt[:-2]
    return [{"file_name": img_dir, "text": txt}]

def convert_text_2_iamge_csv(args):
    """
    converts json files and embed into corresponding image
    .csv
    """
    df = pd.DataFrame()
    for nft_class in os.listdir(args.text_dir):
        nft_dir = os.path.join(args.text_dir, nft_class)
        for f in os.listdir(nft_dir):
            img_name = f[:-4] + "png"
            img_dir = os.path.join(nft_class, img_name)
            json_dir = os.path.join(nft_dir, f)
            data = load_json(json_dir)
            row = handler(nft_class=nft_class, data=data, img_dir=img_dir, args=args)
            new_df = pd.DataFrame(row)
            df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv("/home/emir/Desktop/dev/datasets/nft_dataset/metadata.csv", index=False)
        



def parseargs():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--text-dir", help="Json dataaset directory")
    parser.add_argument("--images-dir", help="images_dir which prequisite for sync process")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")
    args = parser.parse_args()
    if args.verbose:
            print("Verbose mode is enabled")
    return args

if __name__ == "__main__":
    args = parseargs()
    convert_text_2_iamge_csv(args)