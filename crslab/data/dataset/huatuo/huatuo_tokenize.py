import json
import os
from tqdm import tqdm

from resources import resources
from crslab.config import DATASET_PATH

def _tokenize(tokenize ,str):
    """
    Tokenize the data using the provided tokenizer.
    :param data: The string to tokenize.
    :return: The tokenized data.
    """
    if tokenize == 'pkuseg':
        import pkuseg
        seg = pkuseg.pkuseg()
        tokenized_data = seg.cut(str)
        return tokenized_data
    
def process_data(tokenize):
    """
    Process the data using the provided tokenizer.
    :param data: The string to tokenize.
    :return: The tokenized data.
    """
    resource = resources[tokenize]
    dpath = os.path.join(DATASET_PATH, resource['folder_path'])
    if not os.path.exists(dpath):
        print(f"Directory {dpath} does not exist.")
        input("Press Enter to exit...")
        exit(1)
    data_list = ['train_data.json', 'valid_data.json', 'test_data.json']
    for data_file in tqdm(data_list):
        source_data_path = os.path.join(dpath, "source_"+data_file)
        if not os.path.exists(source_data_path):
            print(f"File {source_data_path} does not exist.")
            input("Press Enter to exit...")
            exit(1)
        with open(source_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            for conv in item['conv']:
                conv['text'] = _tokenize(tokenize, conv['text'])
        out_path = os.path.join(dpath, data_file)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"{data_file } processed and saved to {out_path}.")
        
process_data('pkuseg')
