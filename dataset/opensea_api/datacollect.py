import requests
import json
import pandas as pd
import time

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
    def set_chain_and_address(self, chain, address):
        self.collection_url = self.org.replace("$chain_id", chain)
        self.collection_url = self.collection_url.replace("$address_id", address)
        self.temp_url = self.collection_url
        return self
    def set_identifier(self, identifier):
        self.temp_url = self.temp_url.replace("$identifier", identifier)
        return self
    

def read_api(json_file):
    f = open(json_file)
    return json.load(f)['key']

def parse_traits(trait_composed):
    if trait_composed:
        response = requests.get(trait_composed)
        if len(response.text) > 1 : # to confirm body is not empty
            try:
                trait_composed = json.loads(response.text)['attributes']
            except Exception as e:
                print(e)
            # print(trait_composed)
            return trait_composed
        else:
            print(response.text)
            return None
    else:
        return None

def get_url(url_id:int=0) -> url_parser:
    """
    id=0 : is for collection request, (for chain basically)
    id=1 : is for overall whole nfts
    """
    if url_id == 0:
        start_url = url_parser("https://api.opensea.io/api/v2/collections/$")

        headers = {
            "accept": "application/json",
            "x-api-key": read_api("/Users/emirulurak/Desktop/dev/ozu/openseadata/api_key.json")
        }
        return start_url, headers
    elif url_id == 1:
        start_url = url_parser("https://api.opensea.io/api/v2/chain/$chain_id/contract/$address_id/nfts/$identifier")
        headers = {
                    "accept": "application/json",
                    "x-api-key": read_api("/Users/emirulurak/Desktop/dev/ozu/openseadata/api_key.json")
                }
        return start_url, headers
    elif url_id == 2:
        start_url = url_parser("https://api.opensea.io/api/v2/chain/$chain_id/contract/$address_id/nfts?limit=200")
        headers = {
                    "accept": "application/json",
                    "x-api-key": read_api("/Users/emirulurak/Desktop/dev/ozu/openseadata/api_key.json")
                }
        return start_url, headers

import argparse
def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--collection_name", type=str, required=True)
    return args.parse_args()

def init():
    args = parse_args()
    df = read_collection("../nft_names.csv")
    # for l in range(5, len(df['0'])): # this is deprecated
    for l in range(1):
        # collection = df['0'][l]
        collection = args.collection_name
        connection_url, headers = get_url(url_id=0)
        connection_url.set_collection(collection_name=collection)
        metadata = {}
        metadata[collection] = []
        next_url = None
        current_error_id = None
        unique_identifiers = []
        for _ in range(30):
            print(collection)
            response = requests.get(connection_url.get(), headers=headers)
            result = response.text
            data = json.loads(result)
            try:
                collection_chain = data['contracts'][0]['chain']
                collection_address = data['contracts'][0]['address']
            except Exception as e:
                print(data)
                pass
            print(collection_chain)
            print(collection_address)
            
            try:
                nft_url, headers = get_url(url_id=2)
                if _ != 0:
                    nft_url.set_chain_and_address(collection_chain, collection_address)
                    nft_url.replace_next_id(next_url)
                else:
                    nft_url.set_chain_and_address(collection_chain, collection_address)
                response = requests.get(nft_url.get(), headers=headers)
                data = json.loads(response.text)
                print(nft_url.get())
                for i in range(len(data['nfts'])):
                    time.sleep(0.3)
                    nft_instance = {}
                    identifier = data['nfts'][i]['identifier']
                    nft_instance["identifier"] = identifier
                    rarity_url, headers = get_url(url_id=1)
                    rarity_url = rarity_url.set_chain_and_address(collection_chain, collection_address)
                    rarity_url.set_identifier(identifier)
                    response = requests.get(rarity_url.get(), headers=headers)
                    data_rarity = json.loads(response.text)
                    print(f"identifier: {identifier}")
                    if identifier not in unique_identifiers:
                        unique_identifiers.append(identifier)
                        nft_instance["rarity"] = data_rarity['nft']['rarity']
                        nft_instance["traits"] = data_rarity['nft']['traits']
                        nft_instance['img_url'] = data_rarity['nft']['image_url']
                        print(f"{_}:{i}")
                        metadata[collection].append(nft_instance)
                        if len(metadata[collection]) == 1500:
                            current_error_id = 0
                            break
                    else:
                        print(f"duplicate found: {identifier}")
                        pass
                try:
                    if next_url != data['next']:
                        print(next_url)
                        print(data['next'])
                        next_url = data['next']
                    else:
                        print("next page is same, stopping")
                        current_error_id = 0
                        break
                except Exception as e:
                    current_error_id = 0
                    print(f"{e} bu ne")
                    break
                    
            except KeyError as e:
                print(e)
                
            if current_error_id == 0:
                break

        
        with open(f"/Users/emirulurak/Desktop/dev/ozu/openseadata/dataset/{collection}.json", "w") as outfile: 
            json.dump(metadata, outfile)
    

if __name__ == "__main__":
    init()