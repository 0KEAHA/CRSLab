import json
import os
from tqdm import tqdm

from resources import resources
from crslab.config import DATASET_PATH
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


model_path = "H:\ThesisCode\model\qwen"

def _tokenize(tokenize ,string):
    """
    Tokenize the data using the provided tokenizer.
    :param data: The string to tokenize.
    :return: The tokenized data.
    """
    if tokenize == 'pkuseg':
        import pkuseg
        seg = pkuseg.pkuseg()
        tokenized_data = seg.cut(string)
        return tokenized_data
    elif tokenize == 'qwen':
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            pad_token='<|endoftext|>',
            config=config
        )
        tokens = tokenizer.encode(string, add_special_tokens=False)
        return tokens
    
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
        for item in tqdm(data):
            for conv in item['conv']:
                conv['text'] = _tokenize(tokenize, conv['text'])
        out_path = os.path.join(dpath, data_file)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"{data_file } processed and saved to {out_path}.")
        
# process_data('pkuseg')


#print(_tokenize('qwen', '你好，我是一个AI助手。请问有什么可以帮助你的吗？'))

config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True
        )
tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            config=config
        )
# source_str = "你好，我是一个AI助手。请问有什么可以帮助你的吗？"
# print(tokenizer.encode(source_str, add_special_tokens=False))
# tokenized_output = tokenizer(source_str,padding=True, truncation=True, return_tensors="pt")
# print(tokenized_output['input_ids'])
# print(tokenized_output['attention_mask'])
# print(tokenizer.encode(tokenizer.eos_token))
print(config)
# print(tokenizer.pad_token_id)
# print(tokenizer.eos_token_id)
# print(tokenizer.bos_token_id)
