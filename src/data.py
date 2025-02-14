import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import json

#change path of the pre computed clusters file in your running environment or include new pair. (Only use when meta_branch=True )
env2clusterpath = {'guest':'./data/'}

name_map = {'eurlex4k': 'Eurlex-4K','wiki31k': 'Wiki10-31K', 'amazon670k': 'Amazon-670K', 'wiki500k': 'Wiki-500K',
                'amazon3m':'Amazon-3M', 'lfamazontitles131k':'LF-AmazonTitles-131K',
                'lfwikiseealso320k':'LF-WikiSeeAlso-320k' }
dtype_map = {'float16':torch.float16, 'bfloat16':torch.bfloat16}



def collate(batch):
    '''
    collate function to be used when sparse label format is needed.
    
    '''
    tokens = []
    attention_mask = []
    labels = []
    group_label_ids = []
    for i, (t, m, l, g) in enumerate(batch):
        tokens.append(t)
        attention_mask.append(m)
        l_coo = [(i, lbl) for lbl in l]
        labels += l_coo
        group_label_ids.append(g)
    return (
        torch.utils.data.default_collate(tokens),
        torch.utils.data.default_collate(attention_mask),
        torch.Tensor(labels).to(torch.int32).contiguous(),
        torch.utils.data.default_collate(group_label_ids),
    )


class DataHandler:
    '''
    Handle all the data reading, preprocessing ,dataset, dataloader and other stuff.
    
    '''
    def __init__(self,cfg,path):
        
        self.cfg = cfg
        self.path = path
        self.device = torch.device(cfg.environment.device)
        self.low_precision_dtype = dtype_map[cfg.training.amp.dtype]
        self.label_map = {}
        self.read_files()
    
        if cfg.model.auxiliary.use_meta_branch:
            self.group_y = self.load_group(cfg.data.dataset)

        
    def load_group(self,dataset):


        
        '''
        loading of precomputed cluster file
        '''
        print('Loading cluster groups')
        cluster_path = env2clusterpath[self.cfg.environment.running_env] + name_map[dataset] + f'/label_group{self.cfg.model.auxiliary.group_y_group}.npy'
        print('cluster path:',cluster_path)
        return np.load(cluster_path, allow_pickle=True)
        
    def read_files(self):
        
        if not self.cfg.data.is_lf_data:
            self.train_raw_texts = self._read_text_files(self.path.train_raw_texts) 
            self.test_raw_texts = self._read_text_files(self.path.test_raw_texts) 

            self.train_labels = self._read_label_files(self.path.train_labels)
            self.test_labels = self._read_label_files(self.path.test_labels)
        else:
            self.train_raw_texts, train_labels = self._read_lf_files(self.path.train_json)
            self.train_labels = train_labels
            self.test_raw_texts, self.test_labels = self._read_lf_files(self.path.test_json)
            if self.cfg.data.augment_label_data:
                label_raw_texts,label_labels = self._read_lf_files(self.path.label_json,label_json=True)
                self.train_raw_texts += label_raw_texts
                self.train_labels += label_labels

        threshold=100
        label_freq = {}
        for labels in self.train_labels:
            for label in labels:
                label_freq[label] = label_freq.get(label, 0) + 1
        head_labels = [k for k in self.label_map.keys() if label_freq.get(k,0)>threshold]
        tail_labels = [k for k in self.label_map.keys() if label_freq.get(k, 0) <= threshold]
        new_label_map = {}


        for i, k in enumerate(sorted(head_labels)):
            new_label_map[k] = i


        for j, k in enumerate(sorted(tail_labels), start=len(head_labels)):
            new_label_map[k] = j

        sorted_labels = dict(sorted(new_label_map.items(), key=lambda item: item[1], reverse=True)) #sorting in descending order

        #reset label_map and assign new indices
        self.label_map = {}
        print(sorted_labels)
        for idx, (label,freq) in enumerate(sorted_labels.items()):
            self.label_map[label] = idx


        #self.label_map = new_label_map
        #self.head_len = len(head_labels)
     
        
        #for i, k in enumerate(sorted(self.label_map.keys())):
         #   self.label_map[k] = i
        
        
    

        
    def _read_text_files(self,filename):
        container = []
        f = open(filename,encoding="utf8")
        for line in f:
            container.append(line.strip())
    
        return container

    def _read_label_files(self,filename):
        container = []
        f = open(filename,encoding="utf8")
        for line in f:
            for l in line.strip().split():
                self.label_map[l] = 0
            container.append(line.strip().split())
            
        return container
    
    def _read_lf_files(self,file,label_json=False):
        text_data = []
        labels = []
        key = 'title' if 'titles' in  self.cfg.data.dataset else 'content'
        if label_json:
            key='title'
        with open(file) as f:
            for i,line in enumerate(f):
                data = json.loads(line)
                text_data.append(data[key])
                if label_json:
                    labels.append([i])
                else:
                    lbls = data['target_ind']
                    for l in lbls:
                        self.label_map[l]=0
                    labels.append(lbls)

        return text_data,labels
        
    
    def getDatasets(self):
        
        group_y = None
        if self.cfg.model.auxiliary.use_meta_branch:
            group_y = self.group_y
            
        train_dset = SimpleDataset(self.cfg,self.train_raw_texts,self.train_labels,self.label_map,group_y,mode='train',task='train')
        test_dset = SimpleDataset(self.cfg,self.test_raw_texts,self.test_labels,self.label_map,None,mode='test')
        train_dset_eval = SimpleDataset(self.cfg,self.train_raw_texts,self.train_labels,self.label_map,None,mode='train')
        
        return train_dset,test_dset,train_dset_eval
    
    
        
    def getDataLoader(self,dset,mode='train'):
        '''
        #currently  separate dataloader for train (sparse labels) and evaluate (dense labels) in order to fasten both process.
        
        '''
        assert mode in ['train','test'], " mode must be either train or test."
        shuffle=False
        batch_size = self.cfg.data.test_batch_size
        if mode == 'train':
            shuffle=True
            batch_size = self.cfg.data.batch_size

        return DataLoader(dset, batch_size=batch_size, num_workers=self.cfg.data.num_workers, pin_memory=True, shuffle=shuffle, collate_fn=collate)


    
    

class SimpleDataset(Dataset):
    
    def __init__(self,cfg,raw_texts,labels,label_map,group_y,mode='train',task='evaluate'):
        super(SimpleDataset).__init__()
        
        self.cfg = cfg
        self.task = task
        self.group_y = group_y
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.encoder.encoder_tokenizer,do_lower_case=True)
        self.cls_token_id = [101]  # [self.tokenizer.cls_token_id]
        self.sep_token_id = [102]  # [self.tokenizer.sep_token_id]
        
        self.raw_text = raw_texts
        self.labels = labels
        self.label_map = label_map
        self.mode = mode
        
        #Only use when meta_branch=True
        if group_y is not None: 
            # group y mode
            self.group_y, self.n_group_y_labels = [], group_y.shape[0]
            self.map_group_y = np.empty(len(label_map), dtype=np.longlong)
            for idx, labels in enumerate(group_y):
                self.group_y.append([])
                for label in labels:
                    if self.cfg.data.is_lf_data:
                        self.group_y[-1].append(label_map[int(label)]) #changed original: int(label)
                    else:
                        self.group_y[-1].append(label_map[label]) #changed original: int(label)
                self.map_group_y[self.group_y[-1]] = idx #check this line
                self.group_y[-1]  = np.array(self.group_y[-1])
            self.group_y = np.array(self.group_y,dtype=object)
        
            for i in range(len(self.map_group_y )):
                val = self.map_group_y[i]
                if val<0 or val>self.n_group_y_labels:
                    self.map_group_y[i] = random.choice(range(self.n_group_y_labels))
                    
                    
    def __len__(self):
        return len(self.raw_text)
    
    def __getitem__(self,idx):
        
        padding_length = 0
        #raw_text = clean_str(self.raw_text[idx])
        raw_text = self.raw_text[idx]
        tokens = self.tokenizer.encode(raw_text, add_special_tokens=False,truncation=True, max_length=self.cfg.data.max_len)
        tokens = tokens[:self.cfg.data.max_len-2]
        tokens = self.cls_token_id +tokens + self.sep_token_id
        
        if len(tokens)<self.cfg.data.max_len:
            padding_length = self.cfg.data.max_len - len(tokens)
        attention_mask = torch.tensor([1] * len(tokens) + [0] * padding_length)
        tokens = torch.tensor(tokens+([0]*padding_length))
        
        labels = [self.label_map[i] for i in self.labels[idx] if i in self.label_map]

        if self.group_y is not None:
            group_labels = self.map_group_y[labels] # list of group labels 
            group_label_ids = torch.zeros(self.n_group_y_labels)
            group_label_ids = group_label_ids.scatter(0, torch.tensor(group_labels),torch.tensor([1.0 for i in group_labels])) #group labels in one-hot format
        else:
            group_label_ids = torch.zeros(10)
        
        return tokens, attention_mask, labels, group_label_ids