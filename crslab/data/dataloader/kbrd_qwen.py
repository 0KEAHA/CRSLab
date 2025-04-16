# @Time   : 2020/11/27
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com

# UPDATE:
# @Time   : 2020/12/2
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com

import torch
from tqdm import tqdm

from crslab.data.dataloader.base import BaseDataLoader
from crslab.data.dataloader.utils import add_start_end_token_idx, padded_tensor, truncate, merge_utt
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

DEBUG = True

class KBRDQwenDataLoader(BaseDataLoader):
    """Dataloader for model KBRD.

    Notes:
        You can set the following parameters in config:

        - ``'context_truncate'``: the maximum length of context.
        - ``'response_truncate'``: the maximum length of response.
        - ``'entity_truncate'``: the maximum length of mentioned entities in context.

        The following values must be specified in ``vocab``:

        - ``'pad'``
        - ``'start'``
        - ``'end'``
        - ``'pad_entity'``

        the above values specify the id of needed special token.

    """

    def __init__(self, opt, dataset, vocab):
        """

        Args:
            opt (Config or dict): config for dataloader or the whole system.
            dataset: data for model.
            vocab (dict): all kinds of useful size, idx and map between token and idx.

        """
        super().__init__(opt, dataset)
        self.pad_token_idx = vocab['pad']
        self.start_token_idx = vocab['start']
        self.end_token_idx = vocab['end']
        self.pad_entity_idx = vocab['pad_entity']
        self.context_truncate = opt.get('context_truncate', None)
        self.response_truncate = opt.get('response_truncate', None)
        self.entity_truncate = opt.get('entity_truncate', None)
        self.max_seq_len = opt.get('max_seq_len', 1024)
        model_path = opt.get('model_path', None)
        self.qwen_config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    config=self.qwen_config
                )

    def rec_process_fn(self):
        #最终返回的是一个列表，列表中的每个元素是一个字典，字典是每一轮推荐者推荐的实体和电影
        augment_dataset = []
        for conv_dict in tqdm(self.dataset):
            if conv_dict['role'] == 'Recommender':
                for movie in conv_dict['items']:
                    augment_conv_dict = {'context_entities': conv_dict['context_entities'], 'item': movie}
                    augment_dataset.append(augment_conv_dict)
        # if DEBUG:
        #     with open('rec_process_fn.txt', 'w') as f:
        #         f.write(str(augment_dataset))
        #         print('rec_process_fn.txt has been written')
        return augment_dataset

    def rec_batchify(self, batch):
        batch_context_entities = []
        batch_movies = []
        for conv_dict in batch:
            batch_context_entities.append(conv_dict['context_entities'])
            batch_movies.append(conv_dict['item'])
        # if DEBUG:
        #     with open('rec_batchify.txt', 'w') as f:
        #         f.write(str({
        #     "context_entities": batch_context_entities,
        #     "item": torch.tensor(batch_movies, dtype=torch.long)
        # }))
        #         print('rec_batchify.txt has been written')
        return {
            "context_entities": batch_context_entities,
            "item": torch.tensor(batch_movies, dtype=torch.long)
        }

    def conv_process_fn(self, *args, **kwargs):
        return self.retain_recommender_target()

    def conv_batchify(self, batch):
        batch_full_texts = []
        batch_context_entities = []
        
        
        for conv_dict in batch:
            context = "\n".join(conv_dict['context_str'])
            full_text = context + "\n" + conv_dict['response_str']
            batch_full_texts.append(full_text)
            batch_context_entities.append(conv_dict['context_entities'])
        tokenized_output = self.tokenizer(
            batch_full_texts,
            padding='longest', # 填充到批次中最长的序列
            truncation=True,   # 截断到 tokenizer 的最大长度 (或 self.max_length)
            max_length=self.max_seq_len, # 明确指定最大长度
            return_tensors="pt" # 返回 PyTorch 张量
        )
        input_ids = tokenized_output['input_ids']
        attention_mask = tokenized_output['attention_mask']
        
        # 3. 创建 labels - 通常是 input_ids 的副本，并将 padding 部分设为 -100
        # 对于 Causal LM (如 GPT)，模型预测下一个 token，所以 labels 就是 input_ids
        labels = input_ids.clone()
        
        # 将 attention_mask 为 0 (即 padding) 的位置对应的 labels 设置为 -100
        labels[attention_mask == 0] = -100
        
        return {
            # "context_tokens": tensor_context_tokens,
            # "response": tensor_response,
            "context_entities": batch_context_entities,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def policy_batchify(self, *args, **kwargs):
        pass
