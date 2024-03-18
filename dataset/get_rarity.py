import os
import json
import argparse
import pandas as pd
import pandas
import csv
from download_dataset import load_json


def parse_arguments():
    args = argparse.ArgumentParser()
    args.add_argument("--folder_dir", type=str, required=True)
    return args.parse_args()



def main():
    args = parse_arguments()
    metadata_csv_dir = "/Users/emirulurak/Desktop/dev/datasets/nft_dataset/metadata_collection.csv"
    df = pd.read_csv(metadata_csv_dir)
    metadata_csv = []
    for index in range(len(df)):
        try:
            file_name = df["file_name"][index]
            data_name = file_name.replace("/train", ".")[:-4].split("/")[-1]
            print(data_name)
            json_dir = file_name.replace("/train", ".")[:-3] + "json"
            json_dir = os.path.join(args.folder_dir, json_dir)
            data = load_json(json_dir)
            metadata_csv.append({"data_name": data_name, "rarity": data[0]['mintObject']['rarity']})
        except KeyError:
            print(f"KEY ERROR !!!!!!!!")
        
    csv_file_path = 'metadata_old_dataset_rarity_new_159.csv'
    fields = ['data_name', 'rarity']
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in metadata_csv:
            writer.writerow(row)

if __name__ == "__main__":
    main() 
    