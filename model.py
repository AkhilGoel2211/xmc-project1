import torch
from torch import nn
from torch_sparse.rewire import FixedFanIn
from transformers import AutoModel, AutoConfig
from torch.utils.data import Dataset, DataLoader

class TransformerEncoder(nn.Module):
    '''
     Custom Transformer Encoder with configurable model components. 
    '''
    def __init__(self, cfg):
        super(TransformerEncoder, self).__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.environment.device)
        self.transformer = self.load_transformer_model(cfg)
        self.pooler = self.create_pooler()
        
    def load_transformer_model(self, cfg):
        """ Load transformer model based on the provided configuration. """
        model_config = AutoConfig.from_pretrained(cfg.model.encoder.encoder_model)
        #model_config.gradient_checkpointing = True # For Gradient checkpointing of Encoder
        model_config.output_hidden_states = True
        try:
            return AutoModel.from_pretrained(
                cfg.model.encoder.encoder_model, 
                add_pooling_layer=False, 
                config=model_config
            ).to(self.device)
        except Exception as e:
            print(f"Failed to load model with pooling layer removed: {e}")
            return AutoModel.from_pretrained(
                cfg.model.encoder.encoder_model, 
                config=model_config
            ).to(self.device)
      
    def forward(self, tokens,masks):
        '''
        Forward pass through transformer and pooling layers. 
        '''
        return self.pooler(self.transformer(tokens,masks),masks).contiguous()
    
    def create_pooler(self):
        '''
         Create a pooling layer based on the configuration.
        '''
        def pool_last_hidden_avg(tf_output, masks):
            last_hidden_state = tf_output['last_hidden_state']
            input_mask_expanded = masks.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_hidden_state = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_hidden_state / sum_mask
        
        def pool_last_nhidden_conlast(tf_output,masks):
            bert_out = tf_output[-1]
            bert_data = [bert_out[-i][:, 0] for i in range(1, self.cfg.model.encoder.feature_layers+1)]
            return torch.cat(bert_data, dim=-1)
            
        if self.cfg.model.encoder.pool_mode == 'last_hidden_avg':
            return pool_last_hidden_avg
        elif self.cfg.model.encoder.pool_mode == 'last_nhidden_conlast':
            return pool_last_nhidden_conlast
        else:
            raise ValueError('Invalid pooling mode specified in the configuration.')
            


class SimpleTModel(nn.Module):
    '''
     A simple transformer model supporting various configurations and enhancements like LoRA and sparse layers.
    '''

    def __init__(self,cfg,path,group_y_labels,head_len):
        super(SimpleTModel,self).__init__()
     
        self.cfg = cfg
        self.path = path
        self.group_y_labels = group_y_labels
        self.device = torch.device(cfg.environment.device)
        self.encoder = TransformerEncoder(cfg)
        self.configure_components(cfg)
        self.use_sparse_layer = cfg.model.ffi.use_sparse_layer
        self.auxloss_scaling = 0
        self.head_len = head_len
        
    
        
        if cfg.model.encoder.use_torch_compile:
            self.encoder = torch.compile(self.encoder)
        #self.head_layer = nn.Linear(self.head_len, cfg.model.head_label)  # Dense layer for head labels
        #self.tail_layer = nn.Linear(cfg.data.num_labels - self.head_len, cfg.model.tail_label_size)  # Sparse layer for tail labels

        self.head_layer = nn.Linear(self.head_len, 10)  # Dense layer for head labels
        self.tail_layer = nn.Linear(cfg.data.num_labels - self.head_len, 5)  # Sparse layer for tail labels
        
            
    def configure_components(self, cfg):
        """ Configure additional components like dropout, linear layers, etc. based on the model configuration. """
        self.dropout = nn.Dropout(cfg.model.encoder.embed_dropout).to(self.device)

        if cfg.model.auxiliary.use_meta_branch:
            self.auxloss_scaling = cfg.model.auxiliary.auxloss_scaling
            self.group_branch = nn.Linear(
                cfg.model.encoder.feature_layers * self.encoder.transformer.config.hidden_size,
                self.group_y_labels
            ).to(self.device)

        if cfg.model.penultimate.use_penultimate_layer:
            self.penultimate = nn.Linear(
                cfg.model.encoder.feature_layers * self.encoder.transformer.config.hidden_size,
                cfg.model.penultimate.penultimate_size
            ).to(self.device)

        if cfg.model.ffi.use_sparse_layer:
            self.configure_sparse_layer(cfg)
        else:
            self.linear = nn.Linear(cfg.model.ffi.input_features, cfg.data.num_labels).to(self.device)
            nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)  # Gaussian initialization
            
        print(f"cfg.model.ffi.input_features={cfg.model.ffi.input_features}")
  
            
    def configure_sparse_layer(self, cfg):
        """ Configure sparse layer for the model. """
        
        self.linear =  FixedFanIn(cfg.model.ffi.input_features, cfg.model.ffi.output_features, fan_in=cfg.model.ffi.fan_in,
                              prune_mode=cfg.model.ffi.prune_mode, init_mode=cfg.model.ffi.growth_init_mode,
                              rewire_threshold=cfg.model.ffi.rewire_threshold, rewire_fraction=cfg.model.ffi.rewire_fraction).to(self.device)
    
    def rewire(self):
        self.linear.rewire()       
        
    def forward(self,tokens,masks):
        ''' Forward pass through the model. '''
            
        out = self.encoder(tokens,masks)
        out = self.dropout(out)
        dtype = out.dtype
        
        branch_out = self.group_branch(out) if self.cfg.model.auxiliary.use_meta_branch else None
        
        if self.cfg.model.penultimate.use_penultimate_layer:
            out = self.penultimate(out).to(dtype)
            
        #out = self.linear(out)
        head_logits = self.head_layer(out)  # Get logits for head labels
        tail_logits = self.tail_layer(out)  # Get logits for tail labels
        out = torch.cat((head_logits, tail_logits), dim=1)  # Merge logits

        return out, branch_out  
    

        
    def param_list(self):
        param_list, param_list_xmc = [], []
        if self.cfg.model.auxiliary.use_meta_branch:
            param_list.append({"params":self.group_branch.parameters(),"lr":self.cfg.training.optimization.meta_lr})
            
  
        optimizer_params_encoder = []
        for n, p in self.encoder.named_parameters():
            if p.requires_grad:
                optimizer_params_encoder.append((n, p))
        
        no_decay_params = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        param_list += [
                {'params': [p for n, p in optimizer_params_encoder if not any(nd in n for nd in no_decay_params)],
                    'weight_decay': self.cfg.training.optimization.wd_encoder, "lr":self.cfg.training.optimization.encoder_lr},
                {'params': [p for n, p in optimizer_params_encoder if any(nd in n for nd in no_decay_params)],
                    "lr":self.cfg.training.optimization.encoder_lr ,'weight_decay': 0.0}]
            
        if self.cfg.model.penultimate.use_penultimate_layer:
            param_list.append({"params":self.penultimate.parameters(),"lr":self.cfg.training.optimization.penultimate_lr})
        
  
        param_list_xmc.append({"params":self.linear.parameters(),"lr":self.cfg.training.optimization.lr,'weight_decay': self.cfg.training.optimization.wd})
        
        
        return param_list, param_list_xmc

