

import os
import sys
import time
from collections import defaultdict


import torch
from loguru import logger
from nltk import ngrams
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from transformers import (AutoConfig,AutoModelForCausalLM, AutoTokenizer, LogitsProcessor,
                          LogitsProcessorList)
from typing import List, Optional

from evaluator.base import BaseEvaluator
from evaluator.utils import nice_report
from .metrics import *
from pretrained_model import PretrainedModelForKBRDQwen


class StandardEvaluator(BaseEvaluator):
    """The evaluator for all kind of model(recommender, conversation, policy)
    
    Args:
        rec_metrics: the metrics to evaluate recommender model, including hit@K, ndcg@K and mrr@K
        dist_set: the set to record dist n-gram
        dist_cnt: the count of dist n-gram evaluation
        gen_metrics: the metrics to evaluate conversational model, including bleu, dist, embedding metrics, f1
        optim_metrics: the metrics to optimize in training
    """

    def __init__(self,PretrainModel, opt, language, tensorboard=False):
        super(StandardEvaluator, self).__init__()
        # rec
        self.opt = opt
        self.rec_metrics = Metrics()
        # gen
        self.dist_set = defaultdict(set)
        self.dist_cnt = 0
        self.gen_metrics = Metrics()
        # optim
        self.optim_metrics = Metrics()
        # tensorboard
        self.tensorboard = tensorboard
        if self.tensorboard:
            self.writer = SummaryWriter(log_dir='runs/' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
            self.reports_name = ['Recommendation Metrics', 'Generation Metrics', 'Optimization Metrics']
            
        self._use_qwen = opt.get('use_qwen', False)
        if self._use_qwen:
            
            self.config = PretrainModel.config
                # 从本地加载分词器
            self.tokenizer = PretrainModel.tokenizer
            
            self.vocab_size = len(self.tokenizer)

            # 从本地加载模型
            self.model = PretrainModel.model
            self.device = PretrainModel.device

        
    def get_tokens(self,text: str) -> List[str]:
        """Helper to get decoded tokens from text using the provided tokenizer."""
        if text is None :
            return []
        # Tokenize and get IDs
        token_ids = self.tokenizer.encode(text, add_special_tokens=False) # Avoid special tokens in metrics usually
        # Convert IDs to token strings
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        # Optional: Filter out specific unwanted tokens (e.g., special markers if not handled by add_special_tokens)
        # tokens = [t for t in tokens if t not in ['<|im_start|>', '<|im_end|>']] # Example filter
        return tokens

        
        
    def get_single_embedding(self, text: str) -> np.ndarray:
        """
        获取单个文本的词元嵌入 (使用最后一层隐藏状态)。

        Args:
            text (str): 需要获取嵌入的单个文本。

        Returns:
            np.ndarray: 文本对应的词元嵌入 NumPy 数组 (形状为 [actual_seq_len, hidden_size], 去除 padding)。
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                text,  # 对于单个文本，直接传递字符串通常也可以，tokenizer 会处理成列表
                padding=True,  # 即使是单个样本，也可能需要padding以匹配模型处理方式
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 关键修改 1: 设置 output_hidden_states=True
            outputs = self.model(**inputs, output_hidden_states=True)

            # 关键修改 2: 从 outputs.hidden_states 元组中获取最后一层
            # outputs.hidden_states 是一个包含所有层输出的元组
            # 最后一个元素 (-1) 就是最后一层的隐藏状态
            # 形状通常是 (batch_size, seq_len, hidden_size)，这里 batch_size 是 1
            last_hidden_states = outputs.hidden_states[-1]

            # 获取 attention_mask: shape (1, seq_len)
            attention_mask = inputs['attention_mask']
            # 计算实际序列长度（排除padding）
            seq_len = attention_mask[0].sum()

            # 提取非 padding 部分的嵌入，并移至 CPU 转为 NumPy
            # [0] 是因为 batch_size 为 1
            token_embeddings = last_hidden_states[0, :seq_len, :].cpu().to(torch.float32).numpy()
            #token_embeddings = np.mean(token_embeddings, axis=0)

        return token_embeddings
        


    def rec_evaluate(self, ranks, label):
        for k in [1, 10, 50]:
            if len(ranks) >= k:
                self.rec_metrics.add(f"hit@{k}", HitMetric.compute(ranks, label, k))
                self.rec_metrics.add(f"ndcg@{k}", NDCGMetric.compute(ranks, label, k))
                self.rec_metrics.add(f"mrr@{k}", MRRMetric.compute(ranks, label, k))

    def gen_evaluate(self, hyp, refs):
        if hyp is not None: # 确保 hyp 不是 None
            # print(f"hyp: {hyp}")
            # print(f"refs: {refs}")
            # sys.exit(0)
            if self._use_qwen:
                hyp_tokens = self.get_tokens(hyp)
                valid_refs = [ref for ref in refs if ref and ref.strip()] # 过滤掉空/空白的参考
                valid_ref_tokens = [self.get_tokens(ref) for ref in valid_refs]
            else:
                hyp_tokens = hyp
                valid_ref_tokens =refs
                
            self.gen_metrics.add("f1", F1Metric.compute(hyp_tokens, valid_ref_tokens))

            if hyp_tokens:
                for k in range(1, 5):
                    if valid_ref_tokens:
                         self.gen_metrics.add(f"bleu@{k}", BleuMetric.compute(hyp_tokens, valid_ref_tokens, k))
                    for token in ngrams(hyp_tokens, k):
                        self.dist_set[f"dist@{k}"].add(token)
                self.dist_cnt += 1
            if self._use_qwen:
                hyp_emb = (self.get_single_embedding(hyp)).tolist()
                valid_ref_embs = [self.get_single_embedding(ref).tolist() for ref in valid_refs]
            else:
                hyp_emb = (self._get_sent_embedding(hyp))
                valid_ref_embs = [(self._get_sent_embedding(ref)) for ref in refs]
            # print(f"type(hyp_emb): {type(hyp_emb)}")
            # print(f"hyp_emb: {hyp_emb}")
            # print(f"type(valid_ref_embs): {type(valid_ref_embs)}")
            # print(f"len(valid_ref_embs): {len(valid_ref_embs)}")
            # print(f"valid_ref_embs: {valid_ref_embs}")
            
            try:
                self.gen_metrics.add('greedy', GreedyMatch.compute(hyp_emb, valid_ref_embs))
            except Exception as e:
                logger.error(f"Error computing greedy metrics for hyp='{hyp}', refs='{valid_refs}'. Error: {type(e)} - {e}")
            try:
                self.gen_metrics.add('EmbeddingAverage', EmbeddingAverage.compute(hyp_emb, valid_ref_embs))
            except Exception as e:
                logger.error(f"Error computing EmbeddingAverage metrics for hyp='{hyp}', refs='{valid_refs}'. Error: {type(e)} - {e}")
            try:
                self.gen_metrics.add('extreme', VectorExtrema.compute(hyp_emb, valid_ref_embs))
            except Exception as e:
                logger.error(f"Error computing extreme metrics for hyp='{hyp}', refs='{valid_ref_embs}'. Error: {type(e)} - {e}")

    def report(self, epoch=-1, mode='test'):
        for k, v in self.dist_set.items():
            self.gen_metrics.add(k, AverageMetric(len(v) / self.dist_cnt))
        reports = [self.rec_metrics.report(), self.gen_metrics.report(), self.optim_metrics.report()]
        if self.tensorboard and mode != 'test':
            for idx, task_report in enumerate(reports):
                for each_metric, value in task_report.items():
                    self.writer.add_scalars(f'{self.reports_name[idx]}/{each_metric}', {mode: value.value()}, epoch)
        logger.info('\n' + nice_report(aggregate_unnamed_reports(reports)))

    def reset_metrics(self):
        # rec
        self.rec_metrics.clear()
        # conv
        self.gen_metrics.clear()
        self.dist_cnt = 0
        self.dist_set.clear()
        # optim
        self.optim_metrics.clear()
