import requests
import json
import pandas as pd


def read_collection(csv_file):
    df = pd.read_csv(csv_file)
    return df

class url_parser():
    def __init__(self, original_url):
        self.org = original_url
    
    def replace_next_id(self, new_next):
        if new_next:
            return self.org + f"&next={new_next}"        
        else:
            return self.org


def read_api(json_file):
    f = open(json_file)
    return json.load(f)['key']

def init():
    start_url = url_parser("https://api.opensea.io/api/v2/collection/cryptopunks/nfts?limit=200")
    headers = {
        "accept": "application/json",
        "x-api-key": read_api("/Users/emirulurak/Desktop/dev/ozu/openseadata/api_key.json")
    }
    temp_id = start_url.replace_next_id(None)
    nft_count = 0 
    for _ in range(15):
        response = requests.get(temp_id, headers=headers)
        result = response.text
        data = json.loads(result)
        next_idx = data['next']
        temp_id = start_url.replace_next_id(next_idx)
        nft_count += len(data['nfts'])
        print(nft_count)
        print(data['nfts'][0]['identifier'])
    print(nft_count)

if __name__ == "__main__":
    init()