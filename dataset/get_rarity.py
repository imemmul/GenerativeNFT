import os
import json
import argparse
import pandas
import csv

def parse_arguments():
    args = argparse.ArgumentParser()
    args.add_argument("--folder_dir", type=str, required=True)
    return args.parse_args()



def main():
    args = parse_arguments()
    collection_names = os.listdir(args.folder_dir)
    metadata_csv = []
    count = 0
    bugged_ones = []
    for col_name in collection_names:
        if os.path.isdir(os.path.join(args.folder_dir, col_name)) and col_name != "Portals" and col_name != "Genesis Genopet":
            count += 1 
            for file_name in os.listdir(os.path.join(args.folder_dir, col_name)):
                print(f"{count}: {len(metadata_csv)}")
                
                col_instance_json = os.path.join(args.folder_dir, col_name, file_name)
                # col_instance_json = "/Volumes/Backup_Plus/NFT_Dataset/degenfatcats/Degen Fat Cat the 10003rd.json"
                try:
                    with open(col_instance_json, 'r', encoding='cp1252') as j:
                        try:
                            contents = json.loads(j.read())
                            try:
                                rarity = contents[0]['mintObject']['rarity']
                            except:
                                bugged_ones.append(col_name)
                                count -= 1
                                pass
                            name = contents[0]['mintObject']['onChainName']
                            img_url = contents[0]['mintObject']['img']
                            metadata_csv.append({'Collection': col_name, 'Rarity': rarity, 'IdentifierName': name, 'img_url': img_url})
                            print(name)
                            if len(metadata_csv) == count * 1500:
                                break
                        except KeyError:
                            print("passed")
                            pass
                except UnicodeDecodeError:
                    with open(col_instance_json, 'r', encoding='utf-8') as j:
                        try:
                            contents = json.loads(j.read())
                            try:
                                rarity = contents[0]['mintObject']['rarity']
                            except:
                                bugged_ones.append(col_name)
                                count -= 1
                                pass
                            name = contents[0]['mintObject']['onChainName']
                            img_url = contents[0]['mintObject']['img']
                            metadata_csv.append({'Collection': col_name, 'Rarity': rarity, 'IdentifierName': name, 'img_url': img_url})
                            print(name)
                            if len(metadata_csv) == count * 1500:
                                break
                        except KeyError:
                            print("passed")
                            pass
                except Exception as e:
                    print(e)
                    break
    print(bugged_ones)

    # Save as CSV
    csv_file_path = 'metadata_old_dataset_rarity.csv'
    fields = ['Collection', 'Rarity', 'IdentifierName', 'img_url']

    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in metadata_csv:
            writer.writerow(row)

if __name__ == "__main__":
    main() 
    