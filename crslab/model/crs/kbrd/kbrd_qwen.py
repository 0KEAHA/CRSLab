# -*- encoding: utf-8 -*-
# @Time    :   2020/12/4
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

# UPDATE
# @Time   : 2020/1/3, 2021/1/4
# @Author : Xiaolei Wang, Yuanhang Zhou
# @email  : wxl1999@foxmail.com, sdzyh002@gmail.com

r"""
KBRD
====
References:
    Chen, Qibin, et al. `"Towards Knowledge-Based Recommender Dialog System."`_ in EMNLP 2019.

.. _`"Towards Knowledge-Based Recommender Dialog System."`:
   https://www.aclweb.org/anthology/D19-1189/

"""

import torch
import json
import torch.nn.functional as F
from loguru import logger
from torch import nn
from torch_geometric.nn import RGCNConv


from crslab.model.base import BaseModel
from crslab.model.utils.functions import edge_to_pyg_format
from crslab.model.utils.modules.attention import SelfAttentionBatch
from crslab.model.utils.modules.transformer import TransformerDecoder, TransformerEncoder
from transformers import (AutoConfig,AutoModelForCausalLM, AutoTokenizer, LogitsProcessor,
                          LogitsProcessorList)


USE_QWEN = True
DEBUG = True


class KBRDBiasLogitsProcessor(LogitsProcessor):
    """
    Custom LogitsProcessor to add KBRD user bias to Qwen's logits during generation[cite: 1].

    Args:
        user_logits_bias (torch.Tensor): Precomputed bias tensor of shape [batch_size, vocab_size][cite: 2].
    """
    def __init__(self, user_logits_bias: torch.Tensor):
        self.user_logits_bias = user_logits_bias

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Adds the user-specific bias to the logits scores at each generation step[cite: 4]."""
        bias = self.user_logits_bias.to(scores.device) # Ensure bias is on the same device
        scores = scores + bias # Add bias
        return scores



class KBRDQwenModel(BaseModel):
    """

    Attributes:
        vocab_size: A integer indicating the vocabulary size.
        pad_token_idx: A integer indicating the id of padding token.
        start_token_idx: A integer indicating the id of start token.
        end_token_idx: A integer indicating the id of end token.
        token_emb_dim: A integer indicating the dimension of token embedding layer.
        pretrain_embedding: A string indicating the path of pretrained embedding.
        n_entity: A integer indicating the number of entities.
        n_relation: A integer indicating the number of relation in KG.
        num_bases: A integer indicating the number of bases.
        kg_emb_dim: A integer indicating the dimension of kg embedding.
        user_emb_dim: A integer indicating the dimension of user embedding.
        n_heads: A integer indicating the number of heads.
        n_layers: A integer indicating the number of layer.
        ffn_size: A integer indicating the size of ffn hidden.
        dropout: A float indicating the dropout rate.
        attention_dropout: A integer indicating the dropout rate of attention layer.
        relu_dropout: A integer indicating the dropout rate of relu layer.
        learn_positional_embeddings: A boolean indicating if we learn the positional embedding.
        embeddings_scale: A boolean indicating if we use the embeddings scale.
        reduction: A boolean indicating if we use the reduction.
        n_positions: A integer indicating the number of position.
        longest_label: A integer indicating the longest length for response generation.
        user_proj_dim: A integer indicating dim to project for user embedding.

    """

    def __init__(self, opt, device, vocab, side_data):
        """

        Args:
            opt (dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to place the data and model.
            vocab (dict): A dictionary record the vocabulary information.
            side_data (dict): A dictionary record the side data.

        """
        self.device = device
        self.gpu = opt.get("gpu", [-1])

        # kg
        self.n_entity = vocab['n_entity']
        entity_kg = side_data['entity_kg']
        self.n_relation = entity_kg['n_relation']
        self.edge_idx, self.edge_type = edge_to_pyg_format(entity_kg['edge'], 'RGCN')
        self.edge_idx = self.edge_idx.to(device)
        self.edge_type = self.edge_type.to(device)
        self.num_bases = opt.get('num_bases', 8)
        self.kg_emb_dim = opt.get('kg_emb_dim', 300)
        self.user_emb_dim = self.kg_emb_dim
        # transformer
        if USE_QWEN:
            self.model_path = "H:\ThesisCode\model\qwen"
            self.max_seq_len = opt.get('max_seq_len', 1024)
            self.max_gen_len = opt.get('max_gen_len', 512)
            # 词表配置在qwen layer中确定
        
        # 词偏置bias
        self.user_proj_dim = opt.get('user_proj_dim', 512)
    
        # switching network
        

        super(KBRDQwenModel, self).__init__(opt, device)

    def build_model(self, *args, **kwargs):
        self._build_kg_layer()
        self._build_recommendation_layer()
        
        if USE_QWEN:
            self._build_qwen_layer()

    def _build_embedding(self):
        if self.pretrain_embedding is not None:
            self.token_embedding = nn.Embedding.from_pretrained(
                torch.as_tensor(self.pretrain_embedding, dtype=torch.float), freeze=False,
                padding_idx=self.pad_token_idx)
        else:
            self.token_embedding = nn.Embedding(self.vocab_size, self.token_emb_dim, self.pad_token_idx)
            nn.init.normal_(self.token_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)
            nn.init.constant_(self.token_embedding.weight[self.pad_token_idx], 0)
        logger.debug('[Build embedding]')

    def _build_kg_layer(self):
        self.kg_encoder = RGCNConv(self.n_entity, self.kg_emb_dim, self.n_relation, num_bases=self.num_bases)
        self.kg_attn = SelfAttentionBatch(self.kg_emb_dim, self.kg_emb_dim)
        logger.debug('[Build kg layer]')

    def _build_recommendation_layer(self):
        self.rec_bias = nn.Linear(self.kg_emb_dim, self.n_entity)
        self.rec_loss = nn.CrossEntropyLoss()
        logger.debug('[Build recommendation layer]')
        
    def _build_qwen_layer(self):
        
        self.qwen_config = AutoConfig.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            # 从本地加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            pad_token='<|endoftext|>',
            config=self.qwen_config
        )

        # 从本地加载模型
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=self.qwen_config,
            trust_remote_code=True,
            torch_dtype="auto"  # 自动选择数据类型
        ).to(self.device)

        # 词表参数设置
        if DEBUG:
            vocab_file = r"H:\ThesisCode\CRSLab\CRSLab\data\dataset\huatuo\qwen\token2id.json"
            with open(vocab_file, 'r', encoding='utf-8') as f:
                token2id = json.load(f)
            self.tok2ind = token2id
            self.ind2tok = {idx: word for word, idx in token2id.items()}
            
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_idx = self.tokenizer.pad_token_id if not self.tokenizer.pad_token_id == None else self.tokenizer.eos_token_id
        
        self.strat_token_idx = self.tokenizer. bos_token_id if not self.tokenizer.bos_token_id == None else self.tokenizer.eos_token_id
        self.end_token_idx = self.tokenizer.eos_token_id
        
        if DEBUG:
            print("vocab_size: ", self.vocab_size)
            print("pad_token_idx: ", self.pad_token_idx)
            print("start_token_idx: ", self.strat_token_idx)
            print("end_token_idx: ", self.end_token_idx)
        
        self.user_proj_1 = nn.Linear(self.user_emb_dim, self.user_proj_dim)
        self.user_proj_2 = nn.Linear(self.user_proj_dim, self.vocab_size)
        
        self.conv_loss = nn.CrossEntropyLoss()
        
        self.register_buffer('START', torch.LongTensor([self.strat_token_idx]))
        
        logger.debug('[Build QWEN conversation layer]')
        


    def encode_user(self, entity_lists, kg_embedding):
        user_repr_list = []
        for entity_list in entity_lists:
            if entity_list is None:
                user_repr_list.append(torch.zeros(self.user_emb_dim, device=self.device))
                continue
            user_repr = kg_embedding[entity_list]
            user_repr = self.kg_attn(user_repr)
            user_repr_list.append(user_repr)
        return torch.stack(user_repr_list, dim=0)  # (bs, dim)

    def recommend(self, batch, mode):
        context_entities, item = batch['context_entities'], batch['item']
        kg_embedding = self.kg_encoder(None, self.edge_idx, self.edge_type)
        user_embedding = self.encode_user(context_entities, kg_embedding)
        scores = F.linear(user_embedding, kg_embedding, self.rec_bias.bias)
        loss = self.rec_loss(scores, item)
        return loss, scores

    def _starts(self, batch_size):
        """Return bsz start tokens."""
        return self.START.detach().expand(batch_size, 1)


    def decode_qwen_forced(self, input_ids,attention_mask, user_embedding, labels):
        self.qwen_model.train()
        outputs = self.qwen_model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = None,
            return_dict = True
        )
        token_logits = outputs.logits
        user_logits = self.user_proj_2(F.relu(self.user_proj_1(user_embedding.to(self.device)))).unsqueeze(1)
        sum_logits = token_logits + user_logits
        loss = self.conv_loss(sum_logits.view(-1, self.vocab_size), labels.view(-1).to(self.device))
        preds = torch.argmax(sum_logits, dim=-1)
        return loss,preds
        
    
    def converse(self, batch, mode):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        context_entities = batch['context_entities']
        kg_embedding = self.kg_encoder(None, self.edge_idx, self.edge_type)
        user_embedding = self.encode_user(context_entities, kg_embedding)
        if mode!= 'test':
            loss, preds = self.decode_qwen_forced(input_ids, attention_mask, user_embedding, labels)
            return loss, preds
        else:
            user_logits_bias = self.user_proj_2(F.relu(self.user_proj_1(user_embedding.to(self.device))))
            bias_processor = KBRDBiasLogitsProcessor(user_logits_bias)
            logits_processor_list = LogitsProcessorList([bias_processor])
            self.qwen_model.eval()
            with torch.no_grad():
                generated_ids = self.qwen_decoder.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    logits_processor=logits_processor_list, 
                    max_new_tokens=self.max_gen_len,
                    eos_token_id=self.end_token_idx,
                    pad_token_id=self.pad_token_idx, 
                )
            return torch.zeros(1, device=self.device), generated_ids

    

    def forward(self, batch, mode, stage):
        if len(self.gpu) >= 2:
            self.edge_idx = self.edge_idx.cuda(torch.cuda.current_device())
            self.edge_type = self.edge_type.cuda(torch.cuda.current_device())
        if stage == "conv":
            return self.converse(batch, mode)
        if stage == "rec":
            return self.recommend(batch, mode)

    def freeze_parameters(self):
        freeze_models = [self.kg_encoder, self.kg_attn, self.rec_bias]
        for model in freeze_models:
            for p in model.parameters():
                p.requires_grad = False