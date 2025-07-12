import json
import os
from tqdm import tqdm
import torch

from resources import resources
from crslab.config import DATASET_PATH
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


model_path = "H:\ThesisCode\model\qwen2.5-0.5B-Instruct"

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

# config = AutoConfig.from_pretrained(
#             model_path,
#             trust_remote_code=True
#         )
# tokenizer = AutoTokenizer.from_pretrained(
#             model_path,
#             padding_side="left",
#             trust_remote_code=True,
#             config=config
#         )

# qwen_model = AutoModelForCausalLM.from_pretrained(
# model_path,
#         config=config,
#         trust_remote_code=True,
#         torch_dtype="auto"  # 自动选择数据类型
#     )

# sample_str = "你好，我是 一个AI助手。  请问有  什么可以帮助你的吗？"
# print("Original string:", sample_str)
# print("Tokenized Output:", tokenizer(sample_str))
# print("Tokenized string:", tokenizer.tokenize(sample_str))
# print("Token IDs:", tokenizer.encode(sample_str, add_special_tokens=False))
# print("Decoded string:", tokenizer.decode(tokenizer.encode(sample_str, add_special_tokens=False)))


# new_vocab_size = len(tokenizer)
# print(tokenizer.vocab_size)
# print(tokenizer.bos_token_id)
# print(tokenizer.eos_token_id)
# print(tokenizer.pad_token_id)
# # print(len(tokenizer.special_tokens_map['additional_special_tokens']))
# # print(tokenizer.special_tokens_map)
# print("Added tokens:", tokenizer.added_tokens_decoder)
# print("<|im_start|> ID:", tokenizer.convert_tokens_to_ids("<|im_start|>"))
# print("<|im_end|> ID:", tokenizer.convert_tokens_to_ids("<|im_end|>"))
# print("Tokenizer vocab size (including added tokens):", len(tokenizer))
# print("Original model embedding size:", qwen_model.get_input_embeddings().weight.shape[0])


# # 验证模型嵌入层大小是否已更新
# print("Resized model embedding size:", qwen_model.get_input_embeddings().weight.shape[0])
# qwen_model.resize_token_embeddings(new_vocab_size) # 调整模型嵌入层大小
# assert qwen_model.get_input_embeddings().weight.shape[0] == new_vocab_size
# # 对于某些架构，还需要检查LM head
# if hasattr(qwen_model.config, "vocab_size") and qwen_model.config.vocab_size != new_vocab_size:
#      print(f"Warning: model.config.vocab_size ({qwen_model.config.vocab_size}) does not match the new vocab size ({new_vocab_size}). This might be okay if the LM head was also resized implicitly or tied.")
# if hasattr(qwen_model, "lm_head") and qwen_model.lm_head is not None:
#      print("LM head output size:", qwen_model.lm_head.weight.shape[0])
#      assert qwen_model.lm_head.weight.shape[0] == new_vocab_size # 确保输出层也调整了

# conversation = [
#     [
#         {"role": "user", "content": "今天的天气怎么样？"},
#         {"role": "assistant", "content": "今天天气晴朗，适合出门活动。"}
#     ],
#     [
#         {"role": "user", "content": "你喜欢什么运动？"},
#         {"role": "assistant", "content": "我喜欢跑步和游泳。"}
#     ]
# ]
# templated_output = tokenizer.apply_chat_template(
#     conversation,
#     add_generation_prompt=False, 
#     tokenize=False,
#     return_tensors="pt",        # 返回 PyTorch 张量
#     add_special_tokens=False, # 添加特殊标记
# )
# print(type(templated_output))
# print(templated_output)
# tokenized_output = tokenizer(
#     templated_output,
#     padding='longest', # 填充到批次中最长的序列
#     truncation=True,   # 截断到 tokenizer 的最大长度 (或 self.max_length)
#     max_length=512, # 明确指定最大长度
#     return_tensors="pt" # 返回 PyTorch 张量
# )
# print(f"type of tokenized_output: {type(tokenized_output)}")
# print(f"tokenized_output: {tokenized_output}")
# input_ids = tokenized_output['input_ids']
# attention_mask = tokenized_output['attention_mask']

# ignore_index = -100 # PyTorch 交叉熵损失默认忽略的 index
# im_start_id = 151644
# im_end_id = 151645
# tok_system_id = 8948
# tok_user_id = 872
# tok_assistant_id = 77091
# tok_lf_id = 198

# labels = input_ids.clone()
# batch_size, seq_len = input_ids.shape

# for i in range(batch_size):
#     current_labels = labels[i]
#     in_assistant_response = False
#     current_labels[:] = ignore_index
#     seq_token_ids = input_ids[i].tolist() 
#     start_marker_length = 0 # How many tokens form the start marker (e.g., <|im_start|> assistant \n)
#     for j in range(seq_len):
#         token_id = seq_token_ids[j]
#         if token_id == im_start_id and j + 1 < seq_len:
#             next_token_id = seq_token_ids[j+1]
#             if next_token_id == tok_assistant_id:
#                 in_assistant_response = True
#                 # The start marker is <|im_start|> assistant \n
#                 if j + 2 < seq_len and seq_token_ids[j+2] == tok_lf_id:
#                     content_start_index = j + 3
#                 assistant_content_start_idx = content_start_index
#             else:
#                  in_assistant_response = False
#         elif token_id == im_end_id:
#             if in_assistant_response:
#                 if assistant_content_start_idx <= j:
#                     original_tokens_segment = input_ids[i, assistant_content_start_idx : j + 1]
#                     current_labels[assistant_content_start_idx : j + 1] = original_tokens_segment
#                 in_assistant_response = False 

# labels[attention_mask == 0] = ignore_index
# print(f"labels: {labels}")

# tokenized_output = tokenizer.apply_chat_template(
#     conversation,
#     add_generation_prompt=False, 
#     tokenize=True,
#     padding='longest',          # 填充到批次中最长的序列
#     truncation=True,            # 截断到 tokenizer 的最大长度
#     max_length=512,# 明确指定最大长度
#     return_tensors="pt",        # 返回 PyTorch 张量
#     return_attention_mask=True
# )
# print(f"type of tokenized_output: {type(tokenized_output)}")
# print(f"tokenized_output: {tokenized_output}")


# batch_text =[
#     "今天天气不错，你觉得呢？",
#     "我最近在学习Python编程，你有什么建议吗？",
# ]
# tokenized_output = tokenizer(
#     batch_text,
#     padding='longest', # 填充到批次中最长的序列
#     truncation=True, # 截断超过最大长度的序列
#     return_tensors="pt" # 返回PyTorch张量
# )
# input_ids = tokenized_output['input_ids']
# attention_mask = tokenized_output['attention_mask']
# labels = input_ids.clone()
# labels[attention_mask == 0] = -100

# qwen_model.train()
# outputs = qwen_model(input_ids, attention_mask=attention_mask, labels=labels)
# loss = outputs.loss
# logits = outputs.logits
# preds = torch.argmax(logits, dim=-1)
# output_strings = tokenizer.batch_decode(preds, skip_special_tokens=True)
# print(loss)
# print(logits.shape)
# print(preds.shape)
# print(output_strings)
with open(r"H:\ThesisCode\CRSLab\CRSLab\data\dataset\huatuo\qwen\train_data.json",'r',encoding='utf-8') as f:
    data = json.load(f)
print(len(data))

