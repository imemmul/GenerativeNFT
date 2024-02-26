import argparse
import os
import json
import requests
from urllib.parse import urlparse
import csv
from PIL import Image
from io import BytesIO

def get_rarity(download_dir):
    valid_list = ["azuki",
                "sappy-seals",
                "killabears",
                "lazy-lions",
                "genuine-undead",
                "genesis-creepz",
                "bastard-gan-punks-v2",
                "pudgypenguins",
                "beanzofficial",
                "ninja-squad-official",
                "azragames-thehopeful",
                "thewarlords",
                "parallel-avatars",
                "pixelmongen1",
                "kanpai-pandas"]
    col_dir = "/Users/emirulurak/Desktop/dev/ozu/openseadata/dataset"
    csv_contents = []
    for col_json in os.listdir(col_dir):
        # print(col_json[:-5])
        if col_json[:-5] in valid_list:
            with open(os.path.join(col_dir, col_json)) as f:
                data = json.loads(f.read())
                col_name = col_json[:-5]
                if len(data[col_name]) == 1500:
                    # print(len(data[col_name][:1500]))
                    for id in range(len(data[col_name])):
                        data_name = col_name + "_" + data[col_name][id]['identifier'] + ".png"
                        if not os.path.exists(os.path.join(download_dir, data_name)):
                            print(data_name)
                            label = data[col_name][id]['rarity']
                            # TODO deal with none urls
                            img_url = data[col_name][id]['img_url']
                            if img_url:
                                try:
                                    response = requests.get(img_url)
                                    if response.status_code == 200:
                                        img = Image.open(BytesIO(response.content))
                                        img_new = img.resize((512, 512))
                                        # with open(os.path.join(download_dir, data_name), 'wb') as img_file:
                                        #     img_file.write(response.content)
                                        img_new.save(os.path.join(download_dir, data_name))
                                except:
                                    print("broken link")
                                csv_contents.append({'data_name': data_name, 'label':label})
                            else:
                                print("None img url")
                        else:
                            print("already")
                elif len(data[col_name]) > 1500:
                    for id in range(len(data[col_name][:1500])):
                        data_name = col_name + "_" + data[col_name][id]['identifier'] + ".png"
                        if not os.path.exists(os.path.join(download_dir, data_name)):
                            print(data_name)
                            label = data[col_name][id]['rarity']
                            img_url = data[col_name][id]['img_url']
                            if img_url:
                                try:
                                    response = requests.get(img_url)
                                    if response.status_code == 200:
                                        img = Image.open(BytesIO(response.content))
                                        img_new = img.resize((512, 512))
                                        img_new.save(os.path.join(download_dir, data_name))
                                        # with open(os.path.join(download_dir, data_name), 'wb') as img_file:
                                        #     img_file.write(response.content)
                                except:
                                    print("broken link")
                                csv_contents.append({'data_name': data_name, 'label':label})
                            else:
                                print("None img url")
                        else:
                            print("already")
                
    fields = ['data_name', 'label']
    csv_file_path = '/Users/emirulurak/Desktop/dev/ozu/openseadata/dataset/rarity_dataset/labels.csv'

    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in csv_contents:
            writer.writerow(row)
    
                    
                    


def load_json(json_dir):
    data = None
    try:
        with open(json_dir, "r", encoding='cp1252') as json_file:
            data = json.load(json_file)
    except UnicodeDecodeError as e:
        print(f"changing decoding")
        with open(json_dir, "r", encoding='utf-8') as json_file:
            data = json.load(json_file)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        return data
    
def download_text(args):
    if args.sync_dataset:
        assert args.images_dir is not None, "images_dir is required"
        for cat_dir in os.listdir(args.images_dir):
            cat_folder = os.path.join(args.images_dir, cat_dir)
            for img_name in os.listdir(cat_folder):
                json_name = img_name[:-3] + "json"
                json_dir = os.path.join(args.json_dir, cat_dir, json_name)
                data = load_json(json_dir)
                if data:
                    text_info = data[0]["mintObject"]["attributes"]
                    os.makedirs(os.path.join(args.output_dir, cat_dir), exist_ok=True)
                    output_json_dir = os.path.join(args.output_dir, cat_dir, json_name)
                    if args.verbose:
                        print(f"Downloading: {img_name} with attributes to {output_json_dir}")
                    data = {"data": text_info}
                    with open(output_json_dir, "w") as json_file:
                        json.dump(data, json_file, indent=4)


def download_dataset(args):
    """
    downloading dataset
    """           
    for cat_dir in os.listdir(args.json_dir):
        cat_nft = os.path.join(args.json_dir, cat_dir)
        if cat_dir == "Portals" or cat_dir == "Genesis Genopet" or cat_dir == ".DS_Store":
            pass
        else:
            amount = 0
            if os.path.exists(os.path.join(args.output_dir, cat_dir)):
                amount = len(os.listdir(os.path.join(args.output_dir, cat_dir)))
            for j in os.listdir(cat_nft):
                j_dir = os.path.join(cat_nft, j)
                if args.verbose:
                    print(f"Reading: {j_dir}")
                data = load_json(j_dir)
                if data:
                    nft_name = data[0]["mintObject"]["title"]
                    image_url = data[0]["mintObject"]["img"]
                    text_info = data[0]["mintObject"]["attributes"]
                    if args.verbose:
                        print(f"Downloading: {nft_name} with {image_url}")
                    if args.download_images:
                        filename = j[:-4] + 'png'
                        download_dir = os.path.join(args.output_dir, cat_dir)
                        os.makedirs(download_dir, exist_ok=True)
                        if amount <= 3000:
                            if os.path.exists(os.path.join(download_dir, filename)):
                                print(f"Already exists: {nft_name}")
                                continue
                            else:
                                response = requests.get(image_url)
                                if response.status_code == 200:
                                    with open(os.path.join(download_dir, filename), 'wb') as img_file:
                                        img_file.write(response.content)
                                    amount += 1
                                    print(f'Saved to {download_dir}{filename} as {amount}th image')
                                else:
                                    print(f'Failed to download image')
                        else:
                            pass
                    else:
                        print(f"Nothing Happened")
                        


def parseargs():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--json-dir", help="Json dataaset directory")
    parser.add_argument("--output-dir", help="Download directory")
    parser.add_argument("--images-dir", help="images_dir which prequisite for sync process")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")
    parser.add_argument("--download-images", action="store_true", help="test or download images")
    parser.add_argument("--download-text", action="store_true", help="test or download texts")
    parser.add_argument("--sync-dataset", action="store_true", help="syncing images and text files")
    parser.add_argument("--rarity", action="store_true")
    args = parser.parse_args()
    if args.verbose:
            print("Verbose mode is enabled")
    return args
        
def main():
    args = parseargs()
    if args.download_images:
        download_dataset(args)
    elif args.download_text:
        download_text(args)
    elif args.rarity:
        get_rarity("/Users/emirulurak/Desktop/dev/ozu/openseadata/dataset/rarity_dataset")

if __name__ == "__main__":
    main()
    