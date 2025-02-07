import os
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD
import transformers

from log import Logger
from model import SimpleTModel
from evaluate import Evaluator
from utils import print_trainable_parameters, EarlyStopping, CosineDecay

from torch_sparse.ops import sparse_hinge_loss

dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}
optimizer_map = {'adam': Adam, 'adamw': AdamW, 'sgd': SGD}

class Runner:
    
    def __init__(self, cfg, path, data_handler):
        self.cfg = cfg
        self.path = path
        self.label_map = data_handler.label_map
        self.device = torch.device(cfg.environment.device)
        self.low_precision_dtype = dtype_map[cfg.training.amp.dtype]
        self.group_y_labels = 0
        if cfg.model.auxiliary.use_meta_branch:
            self.group_y_labels = data_handler.group_y.shape[0]  # number of meta classes

        if cfg.model.ffi.use_sparse_layer:
            self.rewire_end_epoch = int(cfg.training.optimization.epochs * cfg.model.ffi.rewire_end_epoch)
        else:
            self.rewire_end_epoch = cfg.training.optimization.epochs

        self._initialize_settings(cfg, data_handler)

        # Separate head and tail labels for distinct training behavior
        self.head_labels, self.tail_labels = self._separate_head_tail(data_handler)

        # Logging Object
        if cfg.training.verbose.logging:
            self.LOG = Logger(cfg)

        # Keep track of total iterations, epochs, and rewiring spent
        self.total_iter = 1
        self.total_epoch = 1
        self.total_rewiring = 0

    def _initialize_settings(self, cfg, data_handler):
        self.model = SimpleTModel(cfg, self.path, self.group_y_labels)
        param_list, param_list_xmc = self.model.param_list()
        self.param_count = print_trainable_parameters(self.model)

        # Following Renee practice NGAME M1 encoder is used for short-text titles datasets 
        if cfg.model.encoder.use_ngame_encoder_weights:
            self._load_ngame_encoder()

        self.optimizer, self.optimizer_xmc = self._get_optimizers(param_list, param_list_xmc)
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer, self.cfg.training.optimization.lr_scheduler)
        self.lr_scheduler_xmc = self._get_lr_scheduler(self.optimizer_xmc, self.cfg.training.optimization.lr_scheduler_xmc)

        if cfg.model.auxiliary.use_meta_branch:
            self.meta_loss_fn = torch.nn.BCEWithLogitsLoss()

        if self.cfg.model.ffi.use_sparse_layer:
            total_rewire_steps = cfg.training.optimization.training_steps * self.rewire_end_epoch
            self.rewire_scheduler = CosineDecay(cfg.model.ffi.rewire_threshold, total_rewire_steps, eta_min=0.0001)

        # Auxiliary Loss decay parameter
        if self.cfg.model.auxiliary.use_meta_branch:
            total_meta_steps = cfg.training.optimization.training_steps * cfg.model.auxiliary.meta_cutoff_epoch
            self.auxloss_scheduler = CosineDecay(cfg.model.auxiliary.auxloss_scaling, total_meta_steps, eta_min=0)

        self.early_stopping = EarlyStopping(patience=25, delta=5e-5, mode='max')
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=cfg.training.amp.enabled,
            init_scale=2**12 if cfg.training.amp.enabled else 2**16
        )

        # Evaluators initialization
        self.test_evaluator_me = Evaluator(cfg, data_handler.label_map, data_handler.test_labels)  # memory efficient version but supports only Precision@K
        self.test_evaluator_d = Evaluator(cfg, data_handler.label_map, data_handler.test_labels, train_labels=data_handler.train_labels,
                                          mode='debug', filter_path=self.path.filter_labels_test if cfg.data.use_filter_eval else None)

    # Separate head and tail labels based on frequency thresholds
    def _separate_head_tail(self, data_handler):
        """
        Separates head and tail labels based on label frequency.
        Returns two lists: head_labels and tail_labels.
        """
        label_freq = data_handler.label_freq
        threshold = self.cfg.model.label_split.threshold
        head_labels = [label for label, freq in label_freq.items() if freq >= threshold]
        tail_labels = [label for label, freq in label_freq.items() if freq < threshold]
        return head_labels, tail_labels

    def run_one_epoch(self, epoch, train_loader):
        self.model.train()
        epoch_loss = 0
        bar = tqdm(total=len(train_loader))
        bar.set_description(f'{epoch}')
        self.model.zero_grad()

        for i, data in enumerate(train_loader):
            tokens, mask, labels, group_labels = data

            with torch.autocast(device_type='cuda', dtype=self.low_precision_dtype, enabled=self.cfg.training.amp.enabled):
                tokens, mask, labels = tokens.to(self.device), mask.to(self.device), labels.to(self.device)
                group_labels = group_labels.to(self.device) if self.cfg.model.auxiliary.use_meta_branch else None

                logits, group_logits = self.model(tokens, mask)

                # Head and tail label loss computation
                head_loss = self._compute_head_loss(logits, labels)
                tail_loss = self._compute_tail_loss(logits, labels)

                loss = head_loss + tail_loss  # Combine head and tail losses
                loss /= self.cfg.training.optimization.grad_accum_step

            self.scaler.scale(loss).backward()

            if (i + 1) % self.cfg.training.optimization.grad_accum_step == 0:
                self.scaler.step(self.optimizer)
                self.scaler.step(self.optimizer_xmc) 
                self.scaler.update()
                self.lr_scheduler.step()
                self.lr_scheduler_xmc.step()

                self.optimizer.zero_grad(set_to_none=True)
                self.optimizer_xmc.zero_grad(set_to_none=True)

                if self.cfg.model.ffi.use_sparse_layer and self.cfg.model.ffi.use_rewire_scheduling:
                    self.rewire_scheduler.step()
                    self.model.linear.rewire_threshold = self.rewire_scheduler.get_dr()

                if self.cfg.model.auxiliary.use_meta_branch:
                    self.auxloss_scheduler.step()
                    self.model.auxloss_scaling = self.auxloss_scheduler.get_dr()

            epoch_loss += loss.item()
            if self.cfg.training.verbose.logging:
                self.LOG.iter_loss.append(loss.item())

            if self.cfg.model.ffi.use_sparse_layer:
                # No rewiring happens after self.cfg.rewire_end_epoch 
                if self.total_iter % self.cfg.model.ffi.rewire_interval == 0 and epoch < self.rewire_end_epoch:
                    self.model.rewire()
                    self.total_rewiring += 1

            self.total_iter += 1
            bar.update(1)
            bar.set_postfix(loss=loss.item())
        return epoch_loss / len(train_loader)

    # Head and Tail loss computation
    def _compute_head_loss(self, logits, labels):
        """
        Compute loss for head labels.
        """
        head_mask = torch.isin(labels, torch.tensor(self.head_labels).to(labels.device))
        head_logits = logits * head_mask.float()
        return self._bce_loss(head_logits, labels)

    def _compute_tail_loss(self, logits, labels):
        """
        Compute loss for tail labels.
        """
        tail_mask = torch.isin(labels, torch.tensor(self.tail_labels).to(labels.device))
        tail_logits = logits * tail_mask.float()
        return self._bce_loss(tail_logits, labels)

    def _create_optimizer(self, optimizer_name, param_list):
        default_params = {'momentum': 0.9} if 'adam' not in optimizer_name else {}
        optimizer_class = optimizer_map.get(optimizer_name)
        if optimizer_class is None:
            raise ValueError(f"Unsupported optimizer {optimizer_name}")
        return optimizer_class(param_list, **default_params)

    
    def _get_optimizers(self, param_list, param_list_xmc):
        encoder_optimizer_name = self.cfg.training.optimization.encoder_optimizer
        xmc_optimizer_name = self.cfg.training.optimization.xmc_optimizer
        optimizer = self._create_optimizer(encoder_optimizer_name, param_list)
        optimizer_xmc = self._create_optimizer(xmc_optimizer_name, param_list_xmc)
        return optimizer, optimizer_xmc
    
    def _get_lr_scheduler(self, optimizer,scheduler_type):
        
        if scheduler_type == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                        factor=0.01, patience=10,eps=1e-4, min_lr=1.0e-5, verbose=True)
        elif scheduler_type == "MultiStepLR":
            epochs = self.cfg.training.optimization.epochs
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,gamma=0.05,milestones=[int(epochs/ 2), int(epochs * 3 / 4)],last_epoch=-1)
        elif scheduler_type == "CosineScheduleWithWarmup":
            total_steps = self.cfg.training.optimization.training_steps*self.cfg.training.optimization.epochs
            warmup_steps = self.cfg.training.optimization.warmup_steps
            scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,num_training_steps=total_steps)
        else:
            raise ValueError(f"Unsupported scheduler type {scheduler_type}")
        return scheduler
    
    def run_train(self,train_loader,test_loader,train_loader_eval=None):

        self.best_p1 = self.cfg.training.best_p1

        print('Training Started for a Single Configuration...')
        print(f"Data Config: {self.cfg['data']} Model Config: {self.cfg['model']}  Training Config: {self.cfg['training']}")
        
        if self.cfg.training.verbose.logging:
            self.LOG.initialize_train(self.param_count)
        

        if self.cfg.training.verbose.logging:
            self.LOG.model_memory_logging()
    

        for epoch in range(self.cfg.training.optimization.epochs):
                
            epoch_loss = self.run_one_epoch(epoch,train_loader)
            print(f'Epoch:{epoch+1}   Epoch Loss: {epoch_loss:.7f}')
            
            if self.cfg.training.verbose.logging:
                self.LOG.loss_logging(epoch,epoch_loss)
                self.LOG.naive_memory_logging(epoch)

            #Test Evaluation
            if epoch % self.cfg.training.evaluation.test_evaluate_every==0:
                metrics  = self.test_evaluator_d.Calculate_Metrics(test_loader,self.model)
                if self.cfg.training.verbose.logging:
                    self.LOG.test_perf_logging(epoch,metrics)
                tp1, tp3, tp5 = metrics['P@K'][0], metrics['P@K'][2], metrics['P@K'][4]

                if tp1>self.cfg.training.best_p1 and self.cfg.training.use_checkpoint:
                    self.best_p1 = tp1
                    temp = 'FFI'+self.cfg.model.ffi.fan_in if self.cfg.model.ffi.use_sparse_layer else 'Dense' 
                    temp = temp +'_' + self.cfg.training.checkpoint_file
                    if not os.path.exists(f'models/{self.cfg.data.dataset}/'):
                        os.makedirs(f'models/{self.cfg.data.dataset}/')
                    name = f'models/{self.cfg.data.dataset}/{self.cfg.data.dataset}_{temp}_best_test.pt'
                    self.save_checkpoint(epoch,name)
            
            #Training evaluation 
            if epoch % self.cfg.training.evaluation.train_evaluate_every==0 and self.cfg.training.evaluation.train_evaluate:
                metrics  = self.test_evaluator_me.Calculate_Metrics(train_loader_eval,self.model)
                if self.cfg.training.verbose.logging:
                    self.LOG.train_perf_logging(epoch,metrics)
 

            self.early_stopping(tp3)
            
            if self.early_stopping.early_stop:
                print("Early stopping!")
                break
    
            self.total_epoch += 1
            
            # Logging
            if self.cfg.training.verbose.logging:
                self.LOG.total_epoch += 1
                self.LOG.step(epoch)
                
        print('Training Finished.')
        if self.cfg.training.verbose.logging:
            self.LOG.finalize()
              

    def save_checkpoint(self,epoch,name):
        checkpoint = {
            'config':self.cfg,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, name)

    def load_checkpoint(self,name):
        checkpoint = torch.load(name)
        try:
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        except RuntimeError as E:
            print(E)

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.cfg = checkpoint['config']

        return checkpoint['epoch']

                    
    def feature_extraction(self, data_loader):
        """
        Feature extraction from encoder or penultimate if activated
        
        Args:
        - model: Model for generating predictions.
        - data_loader: DataLoader providing batches of data (tokens, masks, labels).
        - cfg: Configuration object with attributes like num_labels and device.
        
        Returns:
        - A dense numpy array of predictions with shape (N, num_labels), where N is the dataset size.
        """
        self.model.eval()
        pred_list = []  # Use a list to collect arrays
        with torch.no_grad():
            for step, data in tqdm(enumerate(data_loader)):
                tokens, mask, labels, _ = data
                tokens, mask = tokens.to(self.device), mask.to(self.device)
                logits = self.model.encoder(tokens, mask)
                if self.cfg.model.penultimate.use_penultimate_layer:
                    logits = F.relu(self.model.penultimate(logits))
                logits = logits.detach().cpu().numpy()  # Move to CPU and convert to numpy
                pred_list.append(logits)  # Append to the list

        # Concatenate all numpy arrays in the list into a single matrix
        pred_matrix = np.vstack(pred_list)
        
        return pred_matrix
    
    def _load_ngame_encoder(self):
        path_to_ngame_model = self.cfg.model.encoder.ngame_checkpoint
        print("Using NGAME pretrained encoder. Loading from {}".format(path_to_ngame_model))
        new_state_dict = OrderedDict()
        old_state_dict = torch.load(path_to_ngame_model, map_location="cpu")
        for k, v in old_state_dict.items():
            name = k.replace("embedding_labels.encoder.transformer.0.auto_model.", "")
            new_state_dict[name] = v
        new_state_dict.keys()
        print(self.model.encoder.transformer.load_state_dict(new_state_dict, strict=True))
    

    def _bce_loss(self,logits,labels):
        '''
        Sparse BCE loss.
        Ref: https://github.com/microsoft/renee
        '''
        rows,cols = labels[:,0],labels[:,1]
        loss = logits.clamp(min=0.0).sum(dtype=torch.float32) 
        loss -= logits[rows,cols].sum(dtype=torch.float32) 
        loss += (1+(-torch.abs(logits)).exp()).log().sum(dtype=torch.float32)
        return loss
    
    def _loss_calculation(self,logits,labels,group_logits,group_labels,epoch):

        if self.cfg.training.optimization.loss_fn=='squared_hinge':
            loss = sparse_hinge_loss(logits.float(),labels).mean() #/torch.tensor([64.0]).to(self.device)  #changed from mean()

        elif self.cfg.training.optimization.loss_fn== 'bce': 
            loss = self._bce_loss(logits,labels)
            
        else:
            raise ValueError(f"Unsupported loss function type {self.cfg.training.optimization.loss_fn} in config/training/{self.cfg.data.dataset}.yaml file. Use from ['bce','squared_hinge','positive_bce','focal_loss','combined'] ")
        
        if self.cfg.model.auxiliary.use_meta_branch and epoch<self.cfg.model.auxiliary.meta_cutoff_epoch:
            loss = (1-self.model.auxloss_scaling )*loss + self.model.auxloss_scaling*self.meta_loss_fn(group_logits,group_labels)

        return loss




# import os
# import numpy as np
# from collections import OrderedDict
# from tqdm import tqdm
# import torch
# import torch.nn.functional as F
# from torch.optim import Adam,AdamW,SGD
# import transformers

# from log import Logger
# from model import SimpleTModel
# from evaluate import Evaluator
# from utils import print_trainable_parameters, EarlyStopping, CosineDecay

# from torch_sparse.ops import sparse_hinge_loss


# dtype_map = {'float16':torch.float16, 'bfloat16':torch.bfloat16,'float32':torch.float32}
# optimizer_map = {'adam':Adam,'adamw':AdamW,'sgd':SGD}

# class Runner:
    
#     def __init__(self,cfg,path,data_handler):
        
#         self.cfg = cfg
#         self.path = path
#         self.label_map = data_handler.label_map
#         self.device = torch.device(cfg.environment.device)
#         self.low_precision_dtype = dtype_map[cfg.training.amp.dtype]
#         self.group_y_labels = 0
#         if cfg.model.auxiliary.use_meta_branch:
#             self.group_y_labels = data_handler.group_y.shape[0] # number of meta classes
        
#         if cfg.model.ffi.use_sparse_layer:
#             self.rewire_end_epoch = int(cfg.training.optimization.epochs*cfg.model.ffi.rewire_end_epoch)
#         else:
#             self.rewire_end_epoch = cfg.training.optimization.epochs
        
#         self._initialize_settings(cfg,data_handler) 

#         #Logging Object
#         if cfg.training.verbose.logging:
#             self.LOG = Logger(cfg)
        
#         #keep track of total iterations, epochs and rewiring spent
#         self.total_iter = 1
#         self.total_epoch = 1
#         self.total_rewiring = 0
        
        
#     def _initialize_settings(self,cfg,data_handler):
        
#         self.model = SimpleTModel(cfg,self.path,self.group_y_labels)
#         param_list, param_list_xmc = self.model.param_list()
#         self.param_count = print_trainable_parameters(self.model)
        
#         # Following Renee practice NGAME M1 encoder is used for short-text titles datasets 
#         if cfg.model.encoder.use_ngame_encoder_weights:
#             self._load_ngame_encoder()
        
#         self.optimizer, self.optimizer_xmc = self._get_optimizers(param_list,param_list_xmc)
#         self.lr_scheduler = self._get_lr_scheduler(self.optimizer,self.cfg.training.optimization.lr_scheduler)
#         self.lr_scheduler_xmc = self._get_lr_scheduler(self.optimizer_xmc,self.cfg.training.optimization.lr_scheduler_xmc)

            
#         if cfg.model.auxiliary.use_meta_branch:
#             self.meta_loss_fn = torch.nn.BCEWithLogitsLoss()
        
#         if self.cfg.model.ffi.use_sparse_layer:
#             total_rewire_steps = cfg.training.optimization.training_steps*self.rewire_end_epoch
#             self.rewire_scheduler = CosineDecay(cfg.model.ffi.rewire_threshold,total_rewire_steps,eta_min=0.0001)
        
#         #Auxiliary  Loss decay parameter
#         if self.cfg.model.auxiliary.use_meta_branch:
#             total_meta_steps = cfg.training.optimization.training_steps*cfg.model.auxiliary.meta_cutoff_epoch
#             #total_meta_steps = cfg.training.optimization.training_steps*cfg.training.optimization.epochs
#             self.auxloss_scheduler = CosineDecay(cfg.model.auxiliary.auxloss_scaling, total_meta_steps ,eta_min= 0) 
        
#         self.early_stopping = EarlyStopping(patience=25,delta=5e-5,mode='max')
#         self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.amp.enabled, init_scale =2**12 if cfg.training.amp.enabled else 2**16) 
        
#         #Evaluators initialization
#         self.test_evaluator_me = Evaluator(cfg,data_handler.label_map,data_handler.test_labels) #memory efficient version but supports only Precision@K
        
#         self.test_evaluator_d = Evaluator(cfg,data_handler.label_map,data_handler.test_labels,train_labels=data_handler.train_labels,
#                                           mode='debug',filter_path=self.path.filter_labels_test if cfg.data.use_filter_eval else None)
        
      
#     def _create_optimizer(self, optimizer_name, param_list):
#         default_params = {'momentum': 0.9} if 'adam' not in optimizer_name else {}
#         optimizer_class = optimizer_map.get(optimizer_name)
#         if optimizer_class is None:
#             raise ValueError(f"Unsupported optimizer {optimizer_name}")
#         return optimizer_class(param_list, **default_params)

    
#     def _get_optimizers(self, param_list, param_list_xmc):
#         encoder_optimizer_name = self.cfg.training.optimization.encoder_optimizer
#         xmc_optimizer_name = self.cfg.training.optimization.xmc_optimizer
#         optimizer = self._create_optimizer(encoder_optimizer_name, param_list)
#         optimizer_xmc = self._create_optimizer(xmc_optimizer_name, param_list_xmc)
#         return optimizer, optimizer_xmc
    
#     def _get_lr_scheduler(self, optimizer,scheduler_type):
        
#         if scheduler_type == "ReduceLROnPlateau":
#             scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
#                                         factor=0.01, patience=10,eps=1e-4, min_lr=1.0e-5, verbose=True)
#         elif scheduler_type == "MultiStepLR":
#             epochs = self.cfg.training.optimization.epochs
#             scheduler = torch.optim.lr_scheduler.MultiStepLR(
#                 optimizer,gamma=0.05,milestones=[int(epochs/ 2), int(epochs * 3 / 4)],last_epoch=-1)
#         elif scheduler_type == "CosineScheduleWithWarmup":
#             total_steps = self.cfg.training.optimization.training_steps*self.cfg.training.optimization.epochs
#             warmup_steps = self.cfg.training.optimization.warmup_steps
#             scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,num_training_steps=total_steps)
#         else:
#             raise ValueError(f"Unsupported scheduler type {scheduler_type}")
#         return scheduler
        
    
#     def run_one_epoch(self,epoch,train_loader):
#         self.model.train()
#         epoch_loss = 0
#         bar = tqdm(total=len(train_loader))
#         bar.set_description(f'{epoch}')
#         self.model.zero_grad()
#         for i,data in enumerate(train_loader):

#             tokens,mask,labels, group_labels = data
                
#             with torch.autocast(device_type='cuda', dtype=self.low_precision_dtype, enabled=self.cfg.training.amp.enabled):

#                 tokens, mask, labels = tokens.to(self.device),mask.to(self.device), labels.to(self.device)
#                 group_labels = group_labels.to(self.device) if self.cfg.model.auxiliary.use_meta_branch else None
                
#                 logits,group_logits = self.model(tokens,mask)
                
#                 loss = self._loss_calculation(logits,labels,group_logits,group_labels,epoch)
#                 loss /= self.cfg.training.optimization.grad_accum_step
                
#             self.scaler.scale(loss).backward()
            
#             if (i+1) % self.cfg.training.optimization.grad_accum_step==0:
#                 self.scaler.step(self.optimizer)
#                 self.scaler.step(self.optimizer_xmc) 
#                 self.scaler.update()
#                 self.lr_scheduler.step()
#                 self.lr_scheduler_xmc.step()
                    
#                 self.optimizer.zero_grad(set_to_none=True)
#                 self.optimizer_xmc.zero_grad(set_to_none=True)

#                 if self.cfg.model.ffi.use_sparse_layer and self.cfg.model.ffi.use_rewire_scheduling:
#                     self.rewire_scheduler.step()
#                     self.model.linear.rewire_threshold = self.rewire_scheduler.get_dr()
                    
#                 if self.cfg.model.auxiliary.use_meta_branch:
#                     self.auxloss_scheduler.step()
#                     self.model.auxloss_scaling = self.auxloss_scheduler.get_dr()
                
#             epoch_loss += loss.item()
#             if self.cfg.training.verbose.logging:
#                 self.LOG.iter_loss.append(loss.item())

#             if self.cfg.model.ffi.use_sparse_layer:
#                 #No rewiring happens after self.cfg.rewire_end_epoch 
#                 if self.total_iter %self.cfg.model.ffi.rewire_interval==0 and epoch<self.rewire_end_epoch:
#                     self.model.rewire()
#                     self.total_rewiring += 1
                                    
#             self.total_iter +=1
#             bar.update(1)
#             bar.set_postfix(loss=loss.item())        
#         return epoch_loss/ len(train_loader)
        
        
#     def run_train(self,train_loader,test_loader,train_loader_eval=None):

#         self.best_p1 = self.cfg.training.best_p1

#         print('Training Started for a Single Configuration...')
#         print(f"Data Config: {self.cfg['data']} Model Config: {self.cfg['model']}  Training Config: {self.cfg['training']}")
        
#         if self.cfg.training.verbose.logging:
#             self.LOG.initialize_train(self.param_count)
        

#         if self.cfg.training.verbose.logging:
#             self.LOG.model_memory_logging()
    

#         for epoch in range(self.cfg.training.optimization.epochs):
                
#             epoch_loss = self.run_one_epoch(epoch,train_loader)
#             print(f'Epoch:{epoch+1}   Epoch Loss: {epoch_loss:.7f}')
            
#             if self.cfg.training.verbose.logging:
#                 self.LOG.loss_logging(epoch,epoch_loss)
#                 self.LOG.naive_memory_logging(epoch)

#             #Test Evaluation
#             if epoch % self.cfg.training.evaluation.test_evaluate_every==0:
#                 metrics  = self.test_evaluator_d.Calculate_Metrics(test_loader,self.model)
#                 if self.cfg.training.verbose.logging:
#                     self.LOG.test_perf_logging(epoch,metrics)
#                 tp1, tp3, tp5 = metrics['P@K'][0], metrics['P@K'][2], metrics['P@K'][4]

#                 if tp1>self.cfg.training.best_p1 and self.cfg.training.use_checkpoint:
#                     self.best_p1 = tp1
#                     temp = 'FFI'+self.cfg.model.ffi.fan_in if self.cfg.model.ffi.use_sparse_layer else 'Dense' 
#                     temp = temp +'_' + self.cfg.training.checkpoint_file
#                     if not os.path.exists(f'models/{self.cfg.data.dataset}/'):
#                         os.makedirs(f'models/{self.cfg.data.dataset}/')
#                     name = f'models/{self.cfg.data.dataset}/{self.cfg.data.dataset}_{temp}_best_test.pt'
#                     self.save_checkpoint(epoch,name)
            
#             #Training evaluation 
#             if epoch % self.cfg.training.evaluation.train_evaluate_every==0 and self.cfg.training.evaluation.train_evaluate:
#                 metrics  = self.test_evaluator_me.Calculate_Metrics(train_loader_eval,self.model)
#                 if self.cfg.training.verbose.logging:
#                     self.LOG.train_perf_logging(epoch,metrics)
 

#             self.early_stopping(tp3)
            
#             if self.early_stopping.early_stop:
#                 print("Early stopping!")
#                 break
    
#             self.total_epoch += 1
            
#             # Logging
#             if self.cfg.training.verbose.logging:
#                 self.LOG.total_epoch += 1
#                 self.LOG.step(epoch)
                
#         print('Training Finished.')
#         if self.cfg.training.verbose.logging:
#             self.LOG.finalize()
              

#     def save_checkpoint(self,epoch,name):
#         checkpoint = {
#             'config':self.cfg,
#             'state_dict': self.model.state_dict(),
#             'optimizer': self.optimizer.state_dict(),
#             'epoch': epoch
#         }
#         torch.save(checkpoint, name)

#     def load_checkpoint(self,name):
#         checkpoint = torch.load(name)
#         try:
#             self.model.load_state_dict(checkpoint['state_dict'], strict=False)
#         except RuntimeError as E:
#             print(E)

#         self.optimizer.load_state_dict(checkpoint['optimizer'])
#         self.cfg = checkpoint['config']

#         return checkpoint['epoch']

                    
#     def feature_extraction(self, data_loader):
#         """
#         Feature extraction from encoder or penultimate if activated
        
#         Args:
#         - model: Model for generating predictions.
#         - data_loader: DataLoader providing batches of data (tokens, masks, labels).
#         - cfg: Configuration object with attributes like num_labels and device.
        
#         Returns:
#         - A dense numpy array of predictions with shape (N, num_labels), where N is the dataset size.
#         """
#         self.model.eval()
#         pred_list = []  # Use a list to collect arrays
#         with torch.no_grad():
#             for step, data in tqdm(enumerate(data_loader)):
#                 tokens, mask, labels, _ = data
#                 tokens, mask = tokens.to(self.device), mask.to(self.device)
#                 logits = self.model.encoder(tokens, mask)
#                 if self.cfg.model.penultimate.use_penultimate_layer:
#                     logits = F.relu(self.model.penultimate(logits))
#                 logits = logits.detach().cpu().numpy()  # Move to CPU and convert to numpy
#                 pred_list.append(logits)  # Append to the list

#         # Concatenate all numpy arrays in the list into a single matrix
#         pred_matrix = np.vstack(pred_list)
        
#         return pred_matrix
    
#     def _load_ngame_encoder(self):
#         path_to_ngame_model = self.cfg.model.encoder.ngame_checkpoint
#         print("Using NGAME pretrained encoder. Loading from {}".format(path_to_ngame_model))
#         new_state_dict = OrderedDict()
#         old_state_dict = torch.load(path_to_ngame_model, map_location="cpu")
#         for k, v in old_state_dict.items():
#             name = k.replace("embedding_labels.encoder.transformer.0.auto_model.", "")
#             new_state_dict[name] = v
#         new_state_dict.keys()
#         print(self.model.encoder.transformer.load_state_dict(new_state_dict, strict=True))
    

#     def _bce_loss(self,logits,labels):
#         '''
#         Sparse BCE loss.
#         Ref: https://github.com/microsoft/renee
#         '''
#         rows,cols = labels[:,0],labels[:,1]
#         loss = logits.clamp(min=0.0).sum(dtype=torch.float32) 
#         loss -= logits[rows,cols].sum(dtype=torch.float32) 
#         loss += (1+(-torch.abs(logits)).exp()).log().sum(dtype=torch.float32)
#         return loss
    
#     def _loss_calculation(self,logits,labels,group_logits,group_labels,epoch):

#         if self.cfg.training.optimization.loss_fn=='squared_hinge':
#             loss = sparse_hinge_loss(logits.float(),labels).mean() #/torch.tensor([64.0]).to(self.device)  #changed from mean()

#         elif self.cfg.training.optimization.loss_fn== 'bce': 
#             loss = self._bce_loss(logits,labels)
            
#         else:
#             raise ValueError(f"Unsupported loss function type {self.cfg.training.optimization.loss_fn} in config/training/{self.cfg.data.dataset}.yaml file. Use from ['bce','squared_hinge','positive_bce','focal_loss','combined'] ")
        
#         if self.cfg.model.auxiliary.use_meta_branch and epoch<self.cfg.model.auxiliary.meta_cutoff_epoch:
#             loss = (1-self.model.auxloss_scaling )*loss + self.model.auxloss_scaling*self.meta_loss_fn(group_logits,group_labels)

#         return loss