import argparse
import os
import json
import requests
from urllib.parse import urlparse



def download_dataset(args):
    """
    downloading dataset
    """
    for cat_dir in os.listdir(args.json_dir):
        cat_nft = os.path.join(args.json_dir, cat_dir)
        if cat_dir == "Portals":
            pass
        else:
            for j in os.listdir(cat_nft):
                j_dir = os.path.join(cat_nft, j)
                try:
                    with open(j_dir, "r", encoding='cp1252') as json_file:
                        data = json.load(json_file)
                except UnicodeDecodeError as e:
                    print(f"changing decoding")
                    with open(j_dir, "r", encoding='utf-8') as json_file:
                        data = json.load(json_file)
                if data:
                    nft_name = data[0]["mintObject"]["title"]
                    image_url = data[0]["mintObject"]["img"]
                    print(f"Downloading: {nft_name} with {image_url}")
                    if args.download:
                        filename = j[:-4] + 'png'
                        download_dir = os.path.join(args.output_dir, cat_dir)
                        os.makedirs(download_dir, exist_ok=True)
                        response = requests.get(image_url)
                        if response.status_code == 200:
                            with open(os.path.join(download_dir, filename), 'wb') as img_file:
                                img_file.write(response.content)
                            print(f'Saved to {download_dir}{filename}')
                            os.remove(j_dir)
                        else:
                            print(f'Failed to download image')
                        if args.verbose:
                            print(f"Downloading: {nft_name}")
                        


def parseargs():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--json-dir", help="Json dataaset directory")
    parser.add_argument("--output-dir", help="Download directory")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")
    parser.add_argument("--download", action="store_true", help="test or download")
    args = parser.parse_args()
    if args.verbose:
            print("Verbose mode is enabled")
    return args
        
def main():
    args = parseargs()
    download_dataset(args)

if __name__ == "__main__":
    main()
    