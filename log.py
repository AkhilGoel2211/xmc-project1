import datetime
import os
import torch
import wandb
from omegaconf import OmegaConf
import json

class Logger:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.environment.device)
        logfile_path = f'./log' 
        if not os.path.exists(logfile_path ):
            os.mkdir(self.logfile_path)
        self.logfile_name = os.path.join(logfile_path,cfg.training.verbose.log_fname)  #.txtprint(self.logfile_name)
        #self.json_path = os.path.join('Results',cfg.data.dataset,cfg.training.verbose.log_fname)
        #os.makedirs(self.json_path, exist_ok=True)
        #self.json_name = str(cfg.training.seed) + '_'+ str(cfg.job_num)+'.json' if cfg.job_num is not None else str(cfg.training.seed) + '.json'
        #self.json_name = os.path.join(self.json_path,self.json_name)
        #increment scale
        self.total_epoch = 0
        self.total_rewiring = 0
        
        #loss logging
        self.iter_loss, self.epoch_loss = [], []
        
        #for performance logging
        self.trnp1, self.trnp3, self.trnp5 = [], [], []
        self.p1, self.p3, self.p5 = [], [], []
        self.psp1, self.psp3, self.psp5 = [], [], []
        
        # for head and tail label performance tracking
        self.head_p1, self.head_p3, self.head_p5 = [], [], []
        self.tail_p1, self.tail_p3, self.tail_p5 = [], [], []
        self.head_psp1, self.head_psp3, self.head_psp5 = [], [], []
        self.tail_psp1, self.tail_psp3, self.tail_psp5 = [], [], []
        
        self.desc = []  
        
        #memory logging
        self.mem, self.max_mem, self.model_memory = [], [], 0
        self.data = {"Config":{"data":OmegaConf.to_container(cfg.data),"model":OmegaConf.to_container(cfg.model),
                               "training":OmegaConf.to_container(cfg.training)},"Epoch Log":{}}
        #self.logjson()
        
    def initialize_train(self,param_count):
        if self.cfg.training.verbose.logging:
            log_str = f"  Training model for Configuration \n -----Data Config------ \n {OmegaConf.to_yaml(self.cfg['data'])} \n  -----Model Config----- \
                \n {OmegaConf.to_yaml(self.cfg['model'])} \n ------Training Config------ \n {OmegaConf.to_yaml(self.cfg['training'])}"
            self.logfile(log_str)
            self.logfile(param_count)
        if self.cfg.training.verbose.use_wandb:
            wandb.login('allow',"#YOUR WANDB API KEY")
            project_name = self.cfg.dataset.training.verbose.wandb_project_name
            self.run  = wandb.init(project=project_name,config=to_wandb_dict(self.cfg),name=self.cfg.training.verbose.wandb_runname)
                
        
    def logjson(self):
        with open(self.json_name, 'w') as file:
            # Write the updated data structure to the file
            json.dump(self.data, file, indent=4)

    def logfile(self, text):
        with open(self.logfile_name, 'a') as f:
            f.write(datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S') + text + '\n')
         
            
    def naive_memory_logging(self,epoch):
        mem = round(torch.cuda.memory_allocated() / (1024 ** 3), 2)
        max_mem = torch.cuda.max_memory_allocated(device=self.device)
        max_mem = round(max_mem / (1024 ** 3), 2)
        self.mem.append(mem)
        self.max_mem.append(max_mem)
        log_str = f' Memory allocted after trining epoch={epoch} is : {mem} GB \n'
        log_str += f' Peak Memory allocted after trining epoch={epoch} is : {max_mem} GB'
        self.logfile(log_str)
            
    def model_memory_logging(self):
        max_mem = torch.cuda.max_memory_allocated(device=self.device)
        self.model_memory  = round(max_mem / (1024 ** 3), 2)
        self.data["model_memory"] = self.model_memory 
        #self.logjson()
        
    def loss_logging(self,epoch,epoch_loss):
        self.epoch_loss.append(epoch_loss)
        log_str = f'   Epoch: {epoch+1:>2}   train_loss:{epoch_loss}'
        self.logfile(log_str)
        
    def test_perf_logging(self,epoch,metrics):
        
        # Log head and tail label performance if available
        if 'Head_P@K' in metrics and 'Tail_P@K' in metrics:
            self.log_head_tail_performance(epoch, metrics)
        
        tp1, tp3, tp5 = metrics['P@K'][0], metrics['P@K'][2], metrics['P@K'][4]
        self.p1.append(tp1)
        self.p3.append(tp3)
        self.p5.append(tp5)
        log_str = f'  Test set Performance : \n Epoch:{epoch+1:>2}, P@1:{tp1:.5f}   P@3:{tp3:.5f}  P@5:{tp5:.5f}'
        wandb_log = {'Test/P@1':tp1,'Test/P@3':tp3,'Test/P@5':tp5,'Train/Loss':self.epoch_loss[-1]}
        if self.cfg.training.evaluation.eval_psp:
            tpsp1, tpsp3, tpsp5 = metrics['PSP@K'][0], metrics['PSP@K'][2], metrics['PSP@K'][4]
            self.psp1.append(tpsp1)
            self.psp3.append(tpsp3)
            self.psp5.append(tpsp5)
            log_str += f' \n Epoch:{epoch+1:>2}, PSP@1:{tpsp1:.5f}   PSP@3:{tpsp3:.5f}  PSP@5:{tpsp5:.5f}'
            wandb_log.update({'Test/PSP@1':tpsp1,'Test/PSP@3':tpsp3,'Test/PSP@5':tpsp5})
            
        self.logfile(log_str)
            
        if self.cfg.training.verbose.use_wandb:
            wandb_log.update({'epoch':self.total_epoch})
            self.run.log(wandb_log)
            
        print(log_str)
        
def log_head_tail_performance(self, epoch, metrics):
    # Extract head and tail label metrics
    head_p1, head_p3, head_p5 = metrics['Head_P@K'][0], metrics['Head_P@K'][2], metrics['Head_P@K'][4]
    tail_p1, tail_p3, tail_p5 = metrics['Tail_P@K'][0], metrics['Tail_P@K'][2], metrics['Tail_P@K'][4]
    
    self.head_p1.append(head_p1)
    self.head_p3.append(head_p3)
    self.head_p5.append(head_p5)
    self.tail_p1.append(tail_p1)
    self.tail_p3.append(tail_p3)
    self.tail_p5.append(tail_p5)
    
    log_str = f'  Head Label Performance : \n Epoch:{epoch+1:>2}, Head P@1:{head_p1:.5f}   Head P@3:{head_p3:.5f}  Head P@5:{head_p5:.5f}'
    log_str += f'\n  Tail Label Performance : \n Epoch:{epoch+1:>2}, Tail P@1:{tail_p1:.5f}   Tail P@3:{tail_p3:.5f}  Tail P@5:{tail_p5:.5f}'
    
    if 'Head_PSP@K' in metrics and 'Tail_PSP@K' in metrics:
        head_psp1, head_psp3, head_psp5 = metrics['Head_PSP@K'][0], metrics['Head_PSP@K'][2], metrics['Head_PSP@K'][4]
        tail_psp1, tail_psp3, tail_psp5 = metrics['Tail_PSP@K'][0], metrics['Tail_PSP@K'][2], metrics['Tail_PSP@K'][4]
        
        self.head_psp1.append(head_psp1)
        self.head_psp3.append(head_psp3)
        self.head_psp5.append(head_psp5)
        self.tail_psp1.append(tail_psp1)
        self.tail_psp3.append(tail_psp3)
        self.tail_psp5.append(tail_psp5)
        
        log_str += f'\n  Head PSP Label Performance : \n Epoch:{epoch+1:>2}, Head PSP@1:{head_psp1:.5f}   Head PSP@3:{head_psp3:.5f}  Head PSP@5:{head_psp5:.5f}'
        log_str += f'\n  Tail PSP Label Performance : \n Epoch:{epoch+1:>2}, Tail PSP@1:{tail_psp1:.5f}   Tail PSP@3:{tail_psp3:.5f}  Tail PSP@5:{tail_psp5:.5f}'
    
    self.logfile(log_str)
    if self.cfg.training.verbose.use_wandb:
        wandb_log = {
            'Head/Test_P@1': head_p1, 'Head/Test_P@3': head_p3, 'Head/Test_P@5': head_p5,
            'Tail/Test_P@1': tail_p1, 'Tail/Test_P@3': tail_p3, 'Tail/Test_P@5': tail_p5
        }
        if 'Head_PSP@K' in metrics and 'Tail_PSP@K' in metrics:
            wandb_log.update({
                'Head/Test_PSP@1': head_psp1, 'Head/Test_PSP@3': head_psp3, 'Head/Test_PSP@5': head_psp5,
                'Tail/Test_PSP@1': tail_psp1, 'Tail/Test_PSP@3': tail_psp3, 'Tail/Test_PSP@5': tail_psp5
            })
        self.run.log(wandb_log)
    
    def train_perf_logging(self,epoch,metrics):
        trnp1, trnp3, trnp5 = metrics['P@K'][0], metrics['P@K'][2], metrics['P@K'][4]
        self.trnp1.append(trnp1)
        self.trnp3.append(trnp3)
        self.trnp5.append(trnp5)
        print(f"Train set Performance: P@1:{trnp1:.5f}   P@3:{trnp3:.5f}   P@5:{trnp5:.5f}")
        log_str = f" Train set Performance :  Epoch:{epoch+1:>2}   P@1:{trnp1:.5f}   P@3:{trnp3:.5f}  P@5:{trnp5:.5f} "
        wandb_log = {'Train/P@1':trnp1,'Train/P@3':trnp3,'Train/P@5':trnp5}
        self.logfile(log_str)
        if self.cfg.training.verbose.use_wandb:
            wandb_log.update({'epoch':self.total_epoch})
            self.run.log(wandb_log)
            
    
    
    def step(self,epoch):
        
        psp1 = self.psp1[-1] if self.cfg.training.evaluation.eval_psp else 0
        psp3 = self.psp3[-1] if self.cfg.training.evaluation.eval_psp else 0
        psp5 = self.psp5[-1] if self.cfg.training.evaluation.eval_psp else 0
        
        trnp1 = self.trnp1[-1] if self.cfg.training.evaluation.train_evaluate else 0
        trnp3 = self.trnp3[-1] if self.cfg.training.evaluation.train_evaluate else 0
        trnp5 = self.trnp5[-1] if self.cfg.training.evaluation.train_evaluate else 0

        
        
        if self.cfg.model.ffi.use_sparse_layer:

            self.data["Epoch Log"].update({str(epoch):{"trn_loss":self.epoch_loss[-1],"test_P@k":[self.p1[-1],self.p3[-1],self.p5[-1]],
                        "test_PSP@K":[psp1,psp3,psp5],
                        "train_P@k":[trnp1,trnp3,trnp5],
                        "memory":self.mem[-1],"peak_memory":self.max_mem[-1]}})
        else:
            self.data["Epoch Log"].update({str(epoch):{"trn_loss":self.epoch_loss[-1],"test_P@k":[self.p1[-1],self.p3[-1],self.p5[-1]],
                        "test_PSP@K":[psp1,psp3,psp5],
                        "train_P@k":[self.trnp1[-1],self.trnp3[-1],self.trnp5[-1]],
                        "memory":self.mem[-1],"peak_memory":self.max_mem[-1]}})
            
            
        self.data["Epoch Log"].update({str(epoch):{
        "trn_loss": self.epoch_loss[-1],
        "test_P@k": [self.p1[-1], self.p3[-1], self.p5[-1]],
        "test_Head_P@k": [self.head_p1[-1], self.head_p3[-1], self.head_p5[-1]] if self.head_p1 else [0, 0, 0],
        "test_Tail_P@k": [self.tail_p1[-1], self.tail_p3[-1], self.tail_p5[-1]] if self.tail_p1 else [0, 0, 0],
        "memory": self.mem[-1],
        "peak_memory": self.max_mem[-1]
        }})
            
        #self.logjson()
        self._reset_iter_states()
        
    def _reset_iter_states(self):
        self.iter_loss = []

        
    def finalize(self):
        if self.cfg.training.verbose.use_wandb:
            wandb.finish()
        

def to_wandb_dict(cfg):
    '''
    simple fix to create wandb dict
    
    '''
    wandb_cfg = {}
    #Data related config
    wandb_cfg['dataset'] = cfg.data.dataset
    wandb_cfg['num_labels'] = cfg.data.num_labels
    wandb_cfg['max_len'] = cfg.data.max_len
    wandb_cfg['batch_size'] = cfg.data.batch_size
    
    #model related config
    wandb_cfg['encoder_model'] = cfg.model.encoder.encoder_model
    wandb_cfg['embed_dropout'] = cfg.model.encoder.embed_dropout
    wandb_cfg['use_penultimate_layer'] = cfg.model.penultimate.use_penultimate_layer
    wandb_cfg['penultimate_size'] = cfg.model.penultimate.penultimate_size
    wandb_cfg['use_sparse_layer'] = cfg.model.ffi.use_sparse_layer
    wandb_cfg['fan_in'] = cfg.model.ffi.fan_in
    wandb_cfg['prune_mode'] = cfg.model.ffi.prune_mode
    wandb_cfg['rewire_threshold'] = cfg.model.ffi.rewire_threshold
    wandb_cfg['rewire_fraction'] = cfg.model.ffi.rewire_fraction
    wandb_cfg['rewire_interval'] = cfg.model.ffi.rewire_interval
    wandb_cfg['use_meta_branch'] = cfg.model.auxiliary.use_meta_branch
    wandb_cfg['meta_cutoff_epoch'] = cfg.model.auxiliary.meta_cutoff_epoch
    
    #Training related config
    wandb_cfg['loss_fn'] = cfg.training.optimization.loss_fn
    wandb_cfg['encoder_optimizer'] = cfg.training.optimization.encoder_optimizer
    wandb_cfg['xmc_optimizer'] = cfg.training.optimization.xmc_optimizer
    wandb_cfg['epochs'] = cfg.training.optimization.epochs
    wandb_cfg['grad_accum_step'] = cfg.training.optimization.grad_accum_step
    wandb_cfg['encoder_lr'] = cfg.training.optimization.encoder_lr
    wandb_cfg['penultimate_lr'] = cfg.training.optimization.penultimate_lr
    wandb_cfg['meta_lr'] = cfg.training.optimization.meta_lr
    wandb_cfg['lr'] = cfg.training.optimization.lr
    wandb_cfg['wd_encoder'] = cfg.training.optimization.wd_encoder
    wandb_cfg['warmup_steps'] = cfg.training.optimization.warmup_steps
    wandb_cfg['amp_enabled'] = cfg.training.amp.enabled
    wandb_cfg['amp_dtype'] = cfg.training.amp.dtype
    
    return wandb_cfg




# import datetime
# import os
# import torch
# import wandb
# from omegaconf import OmegaConf
# import json

# class Logger:
#     def __init__(self, cfg):
#         self.cfg = cfg
#         self.device = torch.device(cfg.environment.device)
#         logfile_path = f'./log' 
#         if not os.path.exists(logfile_path ):
#             os.mkdir(self.logfile_path)
#         self.logfile_name = os.path.join(logfile_path,cfg.training.verbose.log_fname)  #.txtprint(self.logfile_name)
#         #self.json_path = os.path.join('Results',cfg.data.dataset,cfg.training.verbose.log_fname)
#         #os.makedirs(self.json_path, exist_ok=True)
#         #self.json_name = str(cfg.training.seed) + '_'+ str(cfg.job_num)+'.json' if cfg.job_num is not None else str(cfg.training.seed) + '.json'
#         #self.json_name = os.path.join(self.json_path,self.json_name)
#         #increment scale
#         self.total_epoch = 0
#         self.total_rewiring = 0
        
#         #loss logging
#         self.iter_loss, self.epoch_loss = [], []
        
#         #for performance logging
#         self.trnp1, self.trnp3, self.trnp5 = [], [], []
#         self.p1, self.p3, self.p5 = [], [], []
#         self.psp1, self.psp3, self.psp5 = [], [], []
        
#         self.desc = []  
        
#         #memory logging
#         self.mem, self.max_mem, self.model_memory = [], [], 0
#         self.data = {"Config":{"data":OmegaConf.to_container(cfg.data),"model":OmegaConf.to_container(cfg.model),
#                                "training":OmegaConf.to_container(cfg.training)},"Epoch Log":{}}
#         #self.logjson()
        
#     def initialize_train(self,param_count):
#         if self.cfg.training.verbose.logging:
#             log_str = f"  Training model for Configuration \n -----Data Config------ \n {OmegaConf.to_yaml(self.cfg['data'])} \n  -----Model Config----- \
#                 \n {OmegaConf.to_yaml(self.cfg['model'])} \n ------Training Config------ \n {OmegaConf.to_yaml(self.cfg['training'])}"
#             self.logfile(log_str)
#             self.logfile(param_count)
#         if self.cfg.training.verbose.use_wandb:
#             wandb.login('allow',"#YOUR WANDB API KEY")
#             project_name = self.cfg.dataset.training.verbose.wandb_project_name
#             self.run  = wandb.init(project=project_name,config=to_wandb_dict(self.cfg),name=self.cfg.training.verbose.wandb_runname)
                
        
#     def logjson(self):
#         with open(self.json_name, 'w') as file:
#             # Write the updated data structure to the file
#             json.dump(self.data, file, indent=4)

#     def logfile(self, text):
#         with open(self.logfile_name, 'a') as f:
#             f.write(datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S') + text + '\n')
         
            
#     def naive_memory_logging(self,epoch):
#         mem = round(torch.cuda.memory_allocated() / (1024 ** 3), 2)
#         max_mem = torch.cuda.max_memory_allocated(device=self.device)
#         max_mem = round(max_mem / (1024 ** 3), 2)
#         self.mem.append(mem)
#         self.max_mem.append(max_mem)
#         log_str = f' Memory allocted after trining epoch={epoch} is : {mem} GB \n'
#         log_str += f' Peak Memory allocted after trining epoch={epoch} is : {max_mem} GB'
#         self.logfile(log_str)
            
#     def model_memory_logging(self):
#         max_mem = torch.cuda.max_memory_allocated(device=self.device)
#         self.model_memory  = round(max_mem / (1024 ** 3), 2)
#         self.data["model_memory"] = self.model_memory 
#         #self.logjson()
        
#     def loss_logging(self,epoch,epoch_loss):
#         self.epoch_loss.append(epoch_loss)
#         log_str = f'   Epoch: {epoch+1:>2}   train_loss:{epoch_loss}'
#         self.logfile(log_str)
        
#     def test_perf_logging(self,epoch,metrics):
        
#         tp1, tp3, tp5 = metrics['P@K'][0], metrics['P@K'][2], metrics['P@K'][4]
#         self.p1.append(tp1)
#         self.p3.append(tp3)
#         self.p5.append(tp5)
#         log_str = f'  Test set Performance : \n Epoch:{epoch+1:>2}, P@1:{tp1:.5f}   P@3:{tp3:.5f}  P@5:{tp5:.5f}'
#         wandb_log = {'Test/P@1':tp1,'Test/P@3':tp3,'Test/P@5':tp5,'Train/Loss':self.epoch_loss[-1]}
#         if self.cfg.training.evaluation.eval_psp:
#             tpsp1, tpsp3, tpsp5 = metrics['PSP@K'][0], metrics['PSP@K'][2], metrics['PSP@K'][4]
#             self.psp1.append(tpsp1)
#             self.psp3.append(tpsp3)
#             self.psp5.append(tpsp5)
#             log_str += f' \n Epoch:{epoch+1:>2}, PSP@1:{tpsp1:.5f}   PSP@3:{tpsp3:.5f}  PSP@5:{tpsp5:.5f}'
#             wandb_log.update({'Test/PSP@1':tpsp1,'Test/PSP@3':tpsp3,'Test/PSP@5':tpsp5})
            
#         self.logfile(log_str)
            
#         if self.cfg.training.verbose.use_wandb:
#             wandb_log.update({'epoch':self.total_epoch})
#             self.run.log(wandb_log)
            
#         print(log_str)
    
#     def train_perf_logging(self,epoch,metrics):
#         trnp1, trnp3, trnp5 = metrics['P@K'][0], metrics['P@K'][2], metrics['P@K'][4]
#         self.trnp1.append(trnp1)
#         self.trnp3.append(trnp3)
#         self.trnp5.append(trnp5)
#         print(f"Train set Performance: P@1:{trnp1:.5f}   P@3:{trnp3:.5f}   P@5:{trnp5:.5f}")
#         log_str = f" Train set Performance :  Epoch:{epoch+1:>2}   P@1:{trnp1:.5f}   P@3:{trnp3:.5f}  P@5:{trnp5:.5f} "
#         wandb_log = {'Train/P@1':trnp1,'Train/P@3':trnp3,'Train/P@5':trnp5}
#         self.logfile(log_str)
#         if self.cfg.training.verbose.use_wandb:
#             wandb_log.update({'epoch':self.total_epoch})
#             self.run.log(wandb_log)
            
    
    
#     def step(self,epoch):
        
#         psp1 = self.psp1[-1] if self.cfg.training.evaluation.eval_psp else 0
#         psp3 = self.psp3[-1] if self.cfg.training.evaluation.eval_psp else 0
#         psp5 = self.psp5[-1] if self.cfg.training.evaluation.eval_psp else 0
        
#         trnp1 = self.trnp1[-1] if self.cfg.training.evaluation.train_evaluate else 0
#         trnp3 = self.trnp3[-1] if self.cfg.training.evaluation.train_evaluate else 0
#         trnp5 = self.trnp5[-1] if self.cfg.training.evaluation.train_evaluate else 0

        
        
#         if self.cfg.model.ffi.use_sparse_layer:

#             self.data["Epoch Log"].update({str(epoch):{"trn_loss":self.epoch_loss[-1],"test_P@k":[self.p1[-1],self.p3[-1],self.p5[-1]],
#                         "test_PSP@K":[psp1,psp3,psp5],
#                         "train_P@k":[trnp1,trnp3,trnp5],
#                         "memory":self.mem[-1],"peak_memory":self.max_mem[-1]}})
#         else:
#             self.data["Epoch Log"].update({str(epoch):{"trn_loss":self.epoch_loss[-1],"test_P@k":[self.p1[-1],self.p3[-1],self.p5[-1]],
#                         "test_PSP@K":[psp1,psp3,psp5],
#                         "train_P@k":[self.trnp1[-1],self.trnp3[-1],self.trnp5[-1]],
#                         "memory":self.mem[-1],"peak_memory":self.max_mem[-1]}})
            
#         #self.logjson()
#         self._reset_iter_states()
        
#     def _reset_iter_states(self):
#         self.iter_loss = []

        
#     def finalize(self):
#         if self.cfg.training.verbose.use_wandb:
#             wandb.finish()
        

# def to_wandb_dict(cfg):
#     '''
#     simple fix to create wandb dict
    
#     '''
#     wandb_cfg = {}
#     #Data related config
#     wandb_cfg['dataset'] = cfg.data.dataset
#     wandb_cfg['num_labels'] = cfg.data.num_labels
#     wandb_cfg['max_len'] = cfg.data.max_len
#     wandb_cfg['batch_size'] = cfg.data.batch_size
    
#     #model related config
#     wandb_cfg['encoder_model'] = cfg.model.encoder.encoder_model
#     wandb_cfg['embed_dropout'] = cfg.model.encoder.embed_dropout
#     wandb_cfg['use_penultimate_layer'] = cfg.model.penultimate.use_penultimate_layer
#     wandb_cfg['penultimate_size'] = cfg.model.penultimate.penultimate_size
#     wandb_cfg['use_sparse_layer'] = cfg.model.ffi.use_sparse_layer
#     wandb_cfg['fan_in'] = cfg.model.ffi.fan_in
#     wandb_cfg['prune_mode'] = cfg.model.ffi.prune_mode
#     wandb_cfg['rewire_threshold'] = cfg.model.ffi.rewire_threshold
#     wandb_cfg['rewire_fraction'] = cfg.model.ffi.rewire_fraction
#     wandb_cfg['rewire_interval'] = cfg.model.ffi.rewire_interval
#     wandb_cfg['use_meta_branch'] = cfg.model.auxiliary.use_meta_branch
#     wandb_cfg['meta_cutoff_epoch'] = cfg.model.auxiliary.meta_cutoff_epoch
    
#     #Training related config
#     wandb_cfg['loss_fn'] = cfg.training.optimization.loss_fn
#     wandb_cfg['encoder_optimizer'] = cfg.training.optimization.encoder_optimizer
#     wandb_cfg['xmc_optimizer'] = cfg.training.optimization.xmc_optimizer
#     wandb_cfg['epochs'] = cfg.training.optimization.epochs
#     wandb_cfg['grad_accum_step'] = cfg.training.optimization.grad_accum_step
#     wandb_cfg['encoder_lr'] = cfg.training.optimization.encoder_lr
#     wandb_cfg['penultimate_lr'] = cfg.training.optimization.penultimate_lr
#     wandb_cfg['meta_lr'] = cfg.training.optimization.meta_lr
#     wandb_cfg['lr'] = cfg.training.optimization.lr
#     wandb_cfg['wd_encoder'] = cfg.training.optimization.wd_encoder
#     wandb_cfg['warmup_steps'] = cfg.training.optimization.warmup_steps
#     wandb_cfg['amp_enabled'] = cfg.training.amp.enabled
#     wandb_cfg['amp_dtype'] = cfg.training.amp.dtype
    
#     return wandb_cfg