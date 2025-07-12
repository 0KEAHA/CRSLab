
# system:

def train_conversation(self):
    # if os.environ["CUDA_VISIBLE_DEVICES"] == '-1':
    #     self.model.freeze_parameters()
    # elif len(os.environ["CUDA_VISIBLE_DEVICES"]) == 1:
    #     self.model.freeze_parameters()
    # else:
    #     self.model.module.freeze_parameters()

    if isinstance(self.model, nn.DataParallel):
        model_to_freeze = self.model.module
    else:
        model_to_freeze = self.model
    
    # 调用统一接口
    model_to_freeze.freeze_parameters()
    
    self.init_optim(self.conv_optim_opt, self.model.parameters())
    for epoch in range(self.conv_epoch):
        self.evaluator.reset_metrics()
        logger.info(f'[Conversation epoch {str(epoch)}]')
        logger.info('[Train]')
        for batch in self.train_dataloader.get_conv_data(batch_size=self.conv_batch_size):
            self.step(batch, stage='conv', mode='train')
        self.evaluator.report(epoch=epoch, mode='train')
        # val
        logger.info('[Valid]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.valid_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                self.step(batch, stage='conv', mode='valid')
            self.evaluator.report(epoch=epoch, mode='valid')
            # early stop
            metric = self.evaluator.optim_metrics['gen_loss']
            if self.early_stop(metric):
                break
        # test
        logger.info('[Test]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                self.step(batch, stage='conv', mode='test')
            self.evaluator.report(mode='test')
            
#model

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

    def freeze_parameters(self):
        freeze_models = [self.kg_encoder, self.kg_attn, self.rec_bias]
        for model in freeze_models:
            for p in model.parameters():
                p.requires_grad = False
                
                
# dataloader
def conv_batchify(self, batch):

    batch_context_entities = []
    batch_response_str = []
    batch_full_texts = []
    

    batch_full_conversation = []
    for conv_dict in batch:
        if conv_dict['is_last_round'] == False:
            continue
        batch_response_str.append(conv_dict['response_str'])
        assert(len(conv_dict['context_str']) == len(conv_dict['context_role_list']))
        full_conversation = []
        full_text = ""
        if self.system_message is not None:
            full_conversation.append({"role": "system", "content": self.system_message})
            full_text = full_text + "system\n"+ self.system_message + "\n"
        for i in range(len(conv_dict['context_str'])):
            conv_role =  'user' if conv_dict['context_role_list'][i]=='Seeker' else 'assistant'
            conv_text = conv_dict['context_str'][i]
            full_conversation.append({"role": conv_role, "content": conv_text})
            full_text = full_text + conv_role + "\n" + conv_text + "\n"
        
        resp_role = 'user' if conv_dict['role'] == 'Seeker' else 'assistant'
        resp_text = conv_dict['response_str']
        full_conversation.append({"role": resp_role, "content": resp_text})
        full_text = full_text + resp_role + "\n" + resp_text + "\n"
        batch_full_conversation.append(full_conversation)
        batch_context_entities.append(conv_dict['context_entities'])
        batch_full_texts.append(full_text)

    templated_output = self.tokenizer.apply_chat_template(
        batch_full_conversation, 
        add_generation_prompt=False, 
        tokenize=False,
        return_tensors="pt",        # 返回 PyTorch 张量
        add_special_tokens=False
    )
    tokenized_output = self.tokenizer(
        templated_output,
        padding='longest', # 填充到批次中最长的序列
        truncation=True,   # 截断到 tokenizer 的最大长度 (或 self.max_length)
        max_length=self.max_seq_len, # 明确指定最大长度
        return_tensors="pt" # 返回 PyTorch 张量
    )
    # 在 kbrd_qwen.py 的 conv_batchify 函数中，出错行之前

    input_ids = tokenized_output['input_ids']
    attention_mask = tokenized_output['attention_mask']

    ignore_index = -100 # PyTorch 交叉熵损失默认忽略的 index
    im_start_id = 151644
    im_end_id = 151645
    tok_system_id = 8948
    tok_user_id = 872
    tok_assistant_id = 77091
    tok_lf_id = 198

    labels = input_ids.clone()
    batch_size, seq_len = input_ids.shape

    for i in range(batch_size):
        current_labels = labels[i]
        in_assistant_response = False
        current_labels[:] = ignore_index
        seq_token_ids = input_ids[i].tolist() 
        start_marker_length = 0 # How many tokens form the start marker (e.g., <|im_start|> assistant \n)
        for j in range(seq_len):
            token_id = seq_token_ids[j]
            if token_id == im_start_id and j + 1 < seq_len:
                next_token_id = seq_token_ids[j+1]
                if next_token_id == tok_assistant_id:
                    in_assistant_response = True
                    # The start marker is <|im_start|> assistant \n
                    if j + 2 < seq_len and seq_token_ids[j+2] == tok_lf_id:
                        content_start_index = j + 3
                    assistant_content_start_idx = content_start_index
                else:
                    in_assistant_response = False
            elif token_id == im_end_id:
                if in_assistant_response:
                    if assistant_content_start_idx <= j:
                        original_tokens_segment = input_ids[i, assistant_content_start_idx : j + 1]
                        current_labels[assistant_content_start_idx : j + 1] = original_tokens_segment
                    in_assistant_response = False 

    labels[attention_mask == 0] = ignore_index

    
    return {
        "context_entities": batch_context_entities,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "ground_text":batch_full_texts
    }