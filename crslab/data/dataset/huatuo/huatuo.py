import json
import os
from collections import defaultdict
from copy import copy
import numpy as np
from loguru import logger
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys

from .resources import resources

from crslab.config import DATASET_PATH
from crslab.data.dataset.base import BaseDataset

DEBUG = True



class HuatuoDataset(BaseDataset):
    def __init__(self, opt, tokenize, restore=False, save=False):
        resource = resources[tokenize]
        self.special_token_idx = resource['special_token_idx']
        self.unk_token_idx = self.special_token_idx['unk']
        self.pad_topic_idx = self.special_token_idx['pad_topic']
        dpath = os.path.join(DATASET_PATH, resource['folder_path'], tokenize)
        self.replace_token = opt.get('replace_token',None)
        self.replace_token_idx = opt.get('replace_token_idx',None)
        super().__init__(opt, dpath, resource, restore, save)
    

        
    def _load_data(self):
        train_data, valid_data, test_data = self._load_raw_data()
        self._load_vocab()
        self._load_other_data()

        vocab = {
            'tok2ind': self.tok2ind,
            'ind2tok': self.ind2tok,
            'entity2id': self.entity2id,
            'id2entity': self.id2entity,
            'vocab_size': len(self.tok2ind),
            'n_entity': self.n_entity,
        }
        vocab.update(self.special_token_idx)
        
        # if(DEBUG):
        #     print("train_data: ", train_data)
        #     print("valid_data: ", valid_data)
        #     print("test_data: ", test_data)

        return train_data, valid_data, test_data, vocab
    
    
    def _load_raw_data(self):
        """Load raw data from json files."""
        # train_data
        with open(os.path.join(self.dpath, 'train_data.json'), 'r', encoding='utf-8') as f:
            train_data = json.load(f)
            logger.info(f'[Load train data from {os.path.join(self.dpath, "train_data.json")}]')
        # valid_data
        with open(os.path.join(self.dpath, 'valid_data.json'), 'r', encoding='utf-8') as f:
            valid_data = json.load(f)
            logger.info(f'[Load valid data from {os.path.join(self.dpath, "valid_data.json")}]')
        # test_data
        with open(os.path.join(self.dpath, 'test_data.json'), 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            logger.info(f'[Load test data from {os.path.join(self.dpath, "test_data.json")}]')
        return train_data, valid_data, test_data
    
    def _load_vocab(self):
        self.tok2ind = json.load(open(os.path.join(self.dpath, 'token2id.json'), 'r', encoding='utf-8'))
        self.ind2tok = {idx: word for word, idx in self.tok2ind.items()}
        # add special tokens
        if self.replace_token:
            if self.replace_token not in self.tok2ind:
                if self.replace_token_idx:
                    self.ind2tok[self.replace_token_idx] = self.replace_token
                    self.tok2ind[self.replace_token] = self.replace_token_idx
                    self.special_token_idx[self.replace_token] = self.replace_token_idx
                else:
                    self.ind2tok[len(self.tok2ind)] = self.replace_token
                    self.tok2ind[self.replace_token] = len(self.tok2ind)
                    self.special_token_idx[self.replace_token] = len(self.tok2ind)-1 
        logger.info(f"[Load vocab from {os.path.join(self.dpath, 'token2id.json')}]")
        logger.info(f"[The size of token2index dictionary is {len(self.tok2ind)}]")
        logger.info(f"[The size of index2token dictionary is {len(self.ind2tok)}]")

        
    def _load_other_data(self):
        self.entity2id = json.load(
            open(os.path.join(self.dpath, 'entity2id.json'), encoding='utf-8'))  # {entity: entity_id}
        entityid2order = dict(enumerate(self.entity2id.values()))
        entityid2order = {entity_id: entity for entity, entity_id in entityid2order.items()}
        self.entity2id = {entity: entityid2order[entity_id] for entity, entity_id in self.entity2id.items()}
        self.id2entity = {idx: entity for entity, idx in self.entity2id.items()}
        self.n_entity = len(self.entity2id)
        # entity KG
        self._load_entity_kg()
        
    def _load_entity_kg(self):
        self.entity_kg = pd.read_csv(os.path.join(self.dpath, 'kg_simplified.csv'), header=0, encoding='utf-8')

        
    def _data_preprocess(self,train_data, valid_data, test_data):
        processed_train_data = self._process_conv_data(train_data)
        logger.info(f"[Finish train data process], [The size of train data is {len(processed_train_data)}]")
        processed_valid_data = self._process_conv_data(valid_data)
        logger.info(f"[Finish valid data process], [The size of valid data is {len(processed_valid_data)}]")
        processed_test_data = self._process_conv_data(test_data)
        logger.info(f"[Finish test data process], [The size of test data is {len(processed_test_data)}]")
        processed_side_data = self._process_side_data()
        logger.info(f"[Finish side data process]")
        return processed_train_data, processed_valid_data, processed_test_data, processed_side_data
    
    def _process_conv_data(self, conv_list):
        standard_conv_list = []
        for conv in tqdm(conv_list, desc="Processing conversation data"):
            augmented_conv = self._convert_to_id(conv)
            standard_conv_list.extend(augmented_conv)
        logger.info(f"[The size of standard conversation list is {len(standard_conv_list)}]")
        return standard_conv_list
        
    def _convert_to_id(self, conversation):
        augmented_convs = []
        context_tokens = []
        context_items = []
        for conv in conversation['conv']:
            assert conv['role'].lower() in ['seeker', 'recommender']
            text_token_ids = [self.tok2ind.get(word, self.unk_token_idx) for word in conv["text"]]
            entity_ids = [self.entity2id[entity] for entity in conv["entity"]]
            augmented_convs.append({
                'role': conv['role'].lower().capitalize(),
                'user_profile': None,
                'context_tokens': copy(context_tokens),
                'response': copy(text_token_ids),
                'interaction_history': None,
                'context_items': copy(context_items),
                'items': copy(entity_ids),
                'context_entities': copy(context_items),
                'context_words': None,
                'context_policy': None,
                'target': None,
                'final': None,
            })
            context_tokens.append(text_token_ids)
            context_items.extend(entity_ids)
        return augmented_convs
    
    def _process_side_data(self):
        processed_entity_kg = self._entity_kg_process()
        logger.info(f"[Finish entity KG process], [The size of entity kg is {len(processed_entity_kg)}]")
        
        side_data = {
            "entity_kg": processed_entity_kg,
            "word_kg": None,
            "item_entity_ids": None,
        }
        
        return side_data
    
    def _entity_kg_process(self):
        with open(os.path.join(self.dpath, 'relationid2name.json'), 'r', encoding='utf-8') as f:
            relationid2name = json.load(f)
        relationid2name = {int(k): v for k, v in relationid2name.items()}
        self.entity_kg['relation'] = self.entity_kg['relation'].map(relationid2name)
        name2id = {'症状': 111111, '并发症': 222222}
        self.entity_kg['relation'] = self.entity_kg['relation'].replace(name2id)
        list_of_tuples = list(self.entity_kg.itertuples(index=False, name=None))
        number_relations = (self.entity_kg['relation']).nunique()
        entities = set()
        for h, t, r in list_of_tuples:
            entities.add(self.id2entity.get(h, "unknown"))
            entities.add(self.id2entity.get(t, "unknown"))
            
        return {
            'edge': list_of_tuples,
            'n_relation': number_relations,
            'entity': list(entities),
        }
