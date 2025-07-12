

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
import sys
from loguru import logger
from torch import nn
from torch_geometric.nn import RGCNConv


from model.base import BaseModel
from model.utils.functions import edge_to_pyg_format
from model.utils.modules.attention import SelfAttentionBatch
from model.utils.modules.transformer import TransformerDecoder, TransformerEncoder
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
    def __init__(self, user_logits_bias: torch.Tensor,vocab_size: int,alpha: torch.nn.Parameter):
        self.user_logits_bias = user_logits_bias
        self.vocab_size = vocab_size
        self.alpha = alpha

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Adds the user-specific bias to the logits scores at each generation step[cite: 4]."""
        # print(f"scores.size: {scores.size()}")
        # print(f"user_logits_bias.size: {self.user_logits_bias.size()}")
        # input("Press Enter to continue...")
        bias = self.user_logits_bias.to(scores.device) # Ensure bias is on the same device
        scores = scores[:, :self.vocab_size] # Select the logits for the vocabulary size
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

    def __init__(self, PretrainModel, opt, device, vocab, side_data):
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
        self.use_rgcn = opt.get('use_rgcn', True)
        if self.use_rgcn:
            self.num_rgcn_layers = opt.get('num_rgcn_layers', 1)
            self.rgcn_dropout = opt.get('rgcn_dropout', 0.1)
            self.edge_idx, self.edge_type = edge_to_pyg_format(entity_kg['edge'], 'RGCN')
            self.edge_idx = self.edge_idx.to(device)
            self.edge_type = self.edge_type.to(device)
            self.num_bases = opt.get('num_bases', 8)
        self.kg_emb_dim = opt.get('kg_emb_dim', 300)
        self.user_emb_dim = self.kg_emb_dim
        self.num_neg_samples = opt.get('num_neg_samples', 0)
        self.all_entity_ids_list = list(side_data['item_entity_ids'])
        self.all_entity_ids_set = set(side_data['item_entity_ids'])
        self.mlp_input_dim = self.user_emb_dim + self.kg_emb_dim
        self.mlp_hidden_dim = opt.get("mlp_hidden_dim",self.kg_emb_dim)
        
        if USE_QWEN:
            self.pretrain_model = PretrainModel
            self.model_path = opt.get('model_path',None)
            assert(self.model_path)
            self.max_seq_len = opt.get('max_seq_len', 1024)
            self.max_gen_len = opt.get('max_gen_len', 512)
            # 词表配置在qwen layer中确定
        
        # 词偏置bias
        self.user_proj_dim = opt.get('user_proj_dim', 512)
    
        # switching network
        
        super(KBRDQwenModel, self).__init__(opt, device)

    def build_model(self, *args, **kwargs):
        if self.use_rgcn:
            self._build_kg_layer()
        self._build_recommendation_layer()
        

        self._build_qwen_layer()
    
    def _build_entity_embedding_layer(self):
        """构建实体嵌入层和用户历史实体注意力层"""
        # 使用标准的 nn.Embedding 代替 RGCN
        self.entity_embedding = nn.Embedding(self.n_entity, self.kg_emb_dim)
        # 仍然保留对用户历史实体的自注意力机制，用于聚合信息
        self.entity_attn = SelfAttentionBatch(self.kg_emb_dim, self.kg_emb_dim) # 注意维度匹配
        logger.debug('[Build entity embedding layer and user entity attention layer]')


    def _build_kg_layer(self):
        self.kg_encoder = RGCNConv(self.n_entity, self.kg_emb_dim, self.n_relation, num_bases=self.num_bases)
        self.kg_attn = SelfAttentionBatch(self.kg_emb_dim, self.kg_emb_dim)
        logger.debug('[Build kg layer]')
        self.rgcn_layers = nn.ModuleList()
        for i in range(self.num_rgcn_layers):
            in_channels = self.n_entity if i == 0 else self.kg_emb_dim
            self.rgcn_layers.append(
                RGCNConv(in_channels, self.kg_emb_dim, self.n_relation, num_bases=self.num_bases)
            )
        self.kg_dropout = nn.Dropout(self.rgcn_dropout)
        # self.kg_attn remains the same for user encoding, or could be enhanced too
        self.kg_attn = SelfAttentionBatch(self.kg_emb_dim, self.kg_emb_dim)
        logger.debug(f'[Build {self.num_rgcn_layers}-layer RGCN kg layer]')
    

    def _build_recommendation_layer(self):
        self.rec_bias = nn.Linear(self.kg_emb_dim, self.n_entity)
        self.recommendation_head = nn.Linear(self.user_emb_dim, self.n_entity)
        if self.num_neg_samples == 0:
            self.rec_loss = nn.CrossEntropyLoss()
        else:
            self.rec_loss = nn.BCEWithLogitsLoss()
        # --- Optional: Define an MLP scorer if you don't want simple dot product ---
        self.score_mlp = nn.Sequential(
            nn.Linear(self.mlp_input_dim, self.mlp_hidden_dim),
            nn.ReLU(), # Or nn.LeakyReLU(), nn.Tanh() etc.
            # nn.Dropout(p=0.5), # Optional: Add dropout for regularization
            nn.Linear(self.mlp_hidden_dim, 1) # Output a single score
        )
        
        logger.debug('[Build recommendation layer]')
        
    def _build_qwen_layer(self):
        self.qwen_config = self.pretrain_model.config
        self.tokenizer = self.pretrain_model.tokenizer
        self.vocab_size = len(self.tokenizer)

        self.qwen_model = self.pretrain_model.model.to(self.device)
        # 词表参数
        self.pad_token_idx = self.tokenizer.pad_token_id if not self.tokenizer.pad_token_id == None else self.tokenizer.eos_token_id
        self.strat_token_idx = self.tokenizer.bos_token_id if not self.tokenizer.bos_token_id == None else self.tokenizer.eos_token_id
        self.end_token_idx = self.tokenizer.eos_token_id
        
        self.user_proj_1 = nn.Linear(self.user_emb_dim, self.user_proj_dim)
        self.user_proj_2 = nn.Linear(self.user_proj_dim, self.vocab_size)
        self.alpha = nn.Parameter(torch.tensor(0.0)) # 可学习参数,用于缩放
        self.conv_loss = nn.CrossEntropyLoss(ignore_index=-100)
        
        self.register_buffer('START', torch.LongTensor([self.strat_token_idx]))
        
        logger.debug('[Build QWEN conversation layer]')
        
    def simple_encoder_user(self, entity_lists, all_entity_embeddings):
        user_repr_list = []
        for entity_list in entity_lists:
            if not entity_list: # 检查列表是否为空或 None
                # 如果用户没有历史实体，使用零向量
                user_repr_list.append(torch.zeros(self.user_emb_dim, device=self.device))
                continue
            user_history_embeddings = all_entity_embeddings[entity_list]
            user_repr = self.entity_attn(user_history_embeddings)
            user_repr_list.append(user_repr)
        return torch.stack(user_repr_list, dim=0)  # (bs, user_emb_dim)

    def get_kg_embedding(self):
        # Note: RGCNConv in PyG typically takes node features as input.
        # If using initial node features (e.g., IDs or pretrained), pass them.
        # If starting from scratch, maybe use an initial nn.Embedding or None if the layer handles it.
        # Assuming RGCNConv can handle None input for initial layer (using learnable weights):
        node_features = None # Or nn.Embedding(self.n_entity, self.kg_emb_dim).weight
        for i, layer in enumerate(self.rgcn_layers):
            node_features = layer(node_features, self.edge_idx, self.edge_type)
            if i < self.num_rgcn_layers - 1: # Apply activation and dropout to hidden layers
                node_features = F.relu(node_features)
                node_features = self.kg_dropout(node_features)
        return node_features

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
    
    def simple_recommend(self, batch, mode):
        context_entities, item = batch['context_entities'], batch['item']
        all_entity_embeddings = self.entity_embedding.weight
        user_embedding = self.simple_encoder_user(context_entities, all_entity_embeddings)
        # scores = F.linear(user_embedding, all_entity_embeddings, self.rec_bias.bias)
        scores = self.recommendation_head(user_embedding) # [bs, n_entity]
        loss = self.rec_loss(scores, item)
        return loss, scores
    

    def recommend(self, batch, mode):
        if not self.use_rgcn:
                return self.simple_recommend(batch, mode)
        context_entities, item = batch['context_entities'], batch['item']
        # kg_embedding = self.kg_encoder(None, self.edge_idx, self.edge_type)
        kg_embedding = self.get_kg_embedding()
        user_embedding = self.encode_user(context_entities, kg_embedding)
        scores = F.linear(user_embedding, kg_embedding, self.rec_bias.bias)
        #scores = self.recommendation_head(user_embedding)
        loss = self.rec_loss(scores, item)
        return loss, scores


    def _starts(self, batch_size):
        """Return bsz start tokens."""
        return self.START.detach().expand(batch_size, 1)
    
    def decode_preds(self,preds):
        pred_text = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        pred_text = [text.replace(self.tokenizer.eos_token, '') for text in pred_text]
        pred_text = [text.replace(self.tokenizer.pad_token, '') for text in pred_text]
        return pred_text

    def decode_qwen_forced(self, input_ids,attention_mask, user_embedding, labels):
        self.qwen_model.train()
        outputs = self.qwen_model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels,
            return_dict = True
        )
        token_logits = outputs.logits
        token_logits = token_logits[:, :, :self.vocab_size]
        user_logits = self.user_proj_2(F.relu(self.user_proj_1(user_embedding.to(self.device))))
        scaled_user_logits = self.alpha * user_logits.unsqueeze(1)
        sum_logits = token_logits + scaled_user_logits
        shift_logits = sum_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = self.conv_loss(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
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
            bias_processor = KBRDBiasLogitsProcessor(user_logits_bias,self.vocab_size,self.alpha)
            logits_processor_list = LogitsProcessorList([bias_processor])
            self.qwen_model.eval()
            with torch.no_grad():   
                generated_ids = self.qwen_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    logits_processor=logits_processor_list, 
                    max_new_tokens=self.max_gen_len,
                    eos_token_id = self.end_token_idx,
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