import requests
import json
import pandas as pd


def read_collection(csv_file):
    df = pd.read_csv(csv_file)
    return df

class url_parser():
    def __init__(self, original_url):
        self.org = original_url
        self.collection_url = None
        self.temp_url = None
    
    def replace_next_id(self, new_next):
        if new_next:
            self.temp_url = self.collection_url + f"&next={new_next}"
            return self
        else:
            return self
        
    def set_collection(self, collection_name):
        self.collection_url = self.org.replace("$", collection_name)
        self.temp_url = self.collection_url
        return self
    
    def get(self):
        return self.temp_url

def read_api(json_file):
    f = open(json_file)
    return json.load(f)['key']

def parse_traits(trait_composed):
    response = requests.get(trait_composed)
    trait_composed = json.loads(response.text)['attributes']
    # print(trait_composed)
    return trait_composed
def init():
    start_url = url_parser("https://api.opensea.io/api/v2/collection/$/nfts?limit=200")
    headers = {
        "accept": "application/json",
        "x-api-key": read_api("/Users/emirulurak/Desktop/dev/ozu/openseadata/api_key.json")
    }
    nft_count = 0
    df = read_collection("../nft_names.csv")
    for l in range(len(df['0'])):
        collection = df['0'][l]
        temp_id = start_url.set_collection(collection_name=collection)
        metadata = {}
        for _ in range(15):
            print(collection)
            response = requests.get(temp_id.get(), headers=headers)
            result = response.text
            data = json.loads(result)
            # print(data)
            next_idx = data['next']
            # print(type(data))
            # print(data)
            # print(type(data['nfts']))
            metadata[collection] = []
            for i in range(len(data['nfts'])):
                print(i)
                temp_dict = {}
                temp_dict['collection'] = data['nfts'][i]['collection']
                temp_dict['contract'] = data['nfts'][i]['contract']
                temp_dict['token_standard'] = data['nfts'][i]['token_standard']
                temp_dict['image_url'] = data['nfts'][i]['image_url']
                temp_dict['traits'] = parse_traits(data['nfts'][i]['metadata_url'])
                metadata[collection].append(temp_dict)
                # print(metadata)
            temp_id.replace_next_id(next_idx)
    with open("/Users/emirulurak/Desktop/dev/ozu/openseadata/dataset/metadata.json", "w") as outfile: 
        json.dump(metadata, outfile)
    

if __name__ == "__main__":
    init()