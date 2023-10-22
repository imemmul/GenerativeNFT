import argparse
import os
import json
import requests
from urllib.parse import urlparse

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

if __name__ == "__main__":
    main()
    