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
            print(file_name)
            json_dir = file_name.replace("/train", ".")[:-3] + "json"
            json_dir = os.path.join(args.folder_dir, json_dir)
            data = load_json(json_dir)
            metadata_csv.append({"onChainName": data[0]['mintObject']['onChainName'], "rarity": data[0]['mintObject']['rarity']})
        except KeyError:
            print(f"KEY ERROR !!!!!!!!")
        
    csv_file_path = 'metadata_old_dataset_rarity_new.csv'
    fields = ['onChainName', 'rarity']
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in metadata_csv:
            writer.writerow(row)

if __name__ == "__main__":
    main() 
    