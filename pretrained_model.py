from transformers import (AutoConfig,AutoModelForCausalLM, AutoTokenizer, LogitsProcessor,
                          LogitsProcessorList)
import torch
from loguru import logger


class PretrainedModelForKBRDQwen:
    def __init__(self, opt):
        self.opt = opt
        self.opt = opt
        if opt["gpu"] == [-1]:
            self.device = torch.device('cpu')
        elif len(opt["gpu"]) == 1:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cuda')
        self.model_path = opt.get('model_path',None)
        self.model = None
        self.tokenizer = None
        self.config = None
        
        self._load()
        
    def _load(self):
        self.config = AutoConfig.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            # 从本地加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side='left',
            pad_token='<|endoftext|>',
            config=self.config
        )
        logger.info(f"Load tokenizer from {self.model_path}")
        self.vocab_size = len(self.tokenizer)

        # 从本地加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=self.config,
            trust_remote_code=True,
            torch_dtype='auto'  # 自动选择数据类型
        ).to(self.device)
        logger.info(f"Load model from {self.model_path}")
        self.model.resize_token_embeddings(self.vocab_size)

        # 词表参数
        self.pad_token_idx = self.tokenizer.pad_token_id if not self.tokenizer.pad_token_id == None else self.tokenizer.eos_token_id
        self.strat_token_idx = self.tokenizer.bos_token_id if not self.tokenizer.bos_token_id == None else self.tokenizer.eos_token_id
        self.end_token_idx = self.tokenizer.eos_token_id
        
    def _get_model(self):
        return self.model
    
    def _get_tokenizer(self):
        return self.tokenizer
    
    def _get_config(self):
        return self.config
    
    def _get_vocab_size(self):
        return self.vocab_size
    
    
    