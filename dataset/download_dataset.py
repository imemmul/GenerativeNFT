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
         for j in os.listdir(cat_nft):
            j_dir = os.path.join(cat_nft, j)
            # download jsons to output
            with open(j_dir, "r") as json_file:
                data = json.load(json_file)
            image_url = data.get("image_url") # changeable
            nft_name = data.get("nft_name") # changeable
            filename = j[:-4] + 'jpg'
            download_dir = os.path.join(args.output_dir, cat_dir)
            os.makedirs(download_dir, exist_ok=True)
            response = requests.get(image_url)
            if args.download:
                if response.status_code == 200:
                    with open(os.path.join(download_dir, filename), 'wb') as img_file: # images will be change
                        img_file.write(response.content)
                    print(f'Saved {filename}')
                else:
                    print(f'Failed to download image')
            if args.verbose:
                print(f"Downloading: {nft_name}")


def parseargs():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--json_dir", help="Json dataaset directory")
    parser.add_argument("--output_dir", help="Download directory")
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
    