# -*- coding: utf-8 -*-
import torch.nn as nn
import pickle
import os
import numpy as np
import torch.nn.functional as F
import cv2
import torch
import time
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset,random_split
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import argparse
from utils.params import Params
from torchvision.io import read_video
import random

class Pretrain(nn.Module):
    def __init__(self, input_shape, output_dim):
        super(Pretrain, self).__init__()

        self.input_shape = input_shape
        self.output_dim = output_dim
        self.conv = nn.Conv3d(in_channels = 3, out_channels = 16, kernel_size = (5,5,5))
        self.max_pool = nn.MaxPool3d(kernel_size = (3,3,3))
        self.max_output_dims = [int((self.input_shape[0]-5+1)/3), int((self.input_shape[1]-5+1)/3), int((self.input_shape[2]-5+1)/3)]
        self.batch_norm = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(in_channels = 16, out_channels = 4, kernel_size = (5,5,5))
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.max_output_dims = [int((_dim-5+1)/3) for _dim in self.max_output_dims]
        self.max_output_dim = 4
        for _dim in self.max_output_dims:
            self.max_output_dim = self.max_output_dim*_dim
        self.dense = nn.Linear(in_features = self.max_output_dim,out_features = 300)
        self.dropout = nn.Dropout(0.8)
        self.dense1 = nn.Linear(in_features = 300,out_features = 300)
        self.dense2 = nn.Linear(in_features = 300,out_features = 64)
        self.softmax = nn.Softmax(dim=-1)
        self.dense3 = nn.Linear(in_features = 64, out_features = output_dim)
        
    def get_representation(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        x = self.max_pool(x)
        x = x.reshape(-1, self.max_output_dim)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.sigmoid(x)
        x = self.dense1(x)
        x = self.tanh(x)
        return(x)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        x = self.max_pool(x)
        x = x.reshape(-1, self.max_output_dim)
        x = self.dropout(x)
        x = self.dense(x)
        x = self.sigmoid(x)
        x = self.dense1(x)
        x = self.tanh(x)
        print(x)
        x = self.dense2(x)
        x = self.sigmoid(x)
        x = self.dense3(x)
        x = F.log_softmax(x, dim = -1)
        return(x)
        
def _load_file(file_name):
    feature, label = pickle.load(open(file_name,'rb'))
    return feature, label

class MELDDataset(Dataset):
    def __init__(self, data_dir):
        files = os.listdir(data_dir)
        self.data_files = [os.path.join(data_dir,_file) for _file in files]
        self.data_files = sorted(self.data_files)

    def __getitem__(self, idx):
        return _load_file(self.data_files[idx])
    
    def __len__(self):
        return len(self.data_files)
        
    
def get_raw_feature(video_path):

    vframes, _, _ = read_video(video_path)
    vframes = vframes.permute(3,0,1,2)
    return torch.tensor(vframes/255.0,dtype = torch.float32)

def train(params):
    dataset = MELDDataset(params.stored_path)
    print('training model...')
    total_num = dataset.__len__()
    train_num = int(total_num*params.ratio)
    val_num = total_num - train_num
    train_set, val_set = random_split(dataset, [train_num, val_num])
    train_loader = DataLoader(train_set, batch_size = params.batch_size, shuffle = True)
    val_loader = DataLoader(val_set, batch_size = params.batch_size, shuffle = False)
    output_dim = 7
    

    model = Pretrain((params.input_d, params.input_h, params.input_w), output_dim)
    model = model.to(params.device)
    
    criterion = nn.NLLLoss()
    optimizer = torch.optim.RMSprop(model.parameters(),lr = params.lr)    

    # Temp file for storing the best model 
    epochs = 100
    best_val_loss = 99999.0
#    best_val_loss = -1.0
    for i in range(epochs):
        print('epoch: ', i)
        model.train()
        with tqdm(total = train_num) as pbar:
            time.sleep(0.05)            
            for _i,data in enumerate(train_loader,0):
#                For debugging, please run the line below
#                _i,data = next(iter(enumerate(train_loader,0)))
                b_inputs = data[0].to(params.device)
                b_targets = data[-1].to(params.device)
                
                # Does not train if batch_size is 1, because batch normalization will crash
                if b_inputs[0].shape[0] == 1:
                    continue

                optimizer.zero_grad()
                outputs = model(b_inputs)
                loss = criterion(outputs, b_targets.argmax(dim = -1))
                if np.isnan(loss.item()):
                    print('loss value overflow!')
                    break
                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.8)
                optimizer.step()
                    
                # Compute Training Accuracy                                  
                n_total = len(outputs)
                n_correct = (outputs.argmax(dim = -1) == b_targets.argmax(dim = -1)).sum().item()
                train_acc = n_correct/n_total 

                #Update Progress Bar
                pbar.update(params.batch_size)
                ordered_dict={'acc': train_acc, 'loss':loss.item()}        
                pbar.set_postfix(ordered_dict=ordered_dict)
        
        model.eval()
    
        #Validation Set     
        outputs = []
        targets = []
    
        for _ii,data in enumerate(val_loader,0):  
            data_x = data[0].to(params.device)
            data_t = data[-1].to(params.device)
            data_o = model(data_x)
                            
            outputs.append(data_o.detach())
            targets.append(data_t.detach())
                
        outputs = torch.cat(outputs)
        targets = torch.cat(targets)
        val_loss = criterion(outputs, targets.argmax(dim = -1))
        
        n_total = len(outputs)
        n_correct = (outputs.argmax(dim = -1) == targets.argmax(dim = -1)).sum().item()
        val_acc = n_correct/n_total 
                
        print('val loss = {}, val acc = {}'.format(val_loss,val_acc))
        if val_loss < best_val_loss:
             torch.save(model,'dummy_files/best_model.pt')
             print('The best model up till now. Saved to File.')
             best_val_loss = val_loss
             
    return model

def extract_raw_features(params):
    data = pickle.load(open(params.file_path,'rb'))
    video_ids, video_speakers, video_labels,video_text, video_audio, video_sentence, video_act, train_vid, test_vid, _ = data
    
    if not os.path.exists(params.stored_path):
        os.mkdir(params.stored_path)
        
    train_dia_num = 1038
    val_dia_num = len(train_vid) - train_dia_num 
    
    for dia_id, sen_ids in video_ids.items():
        print('extracting raw features of dia {}'.format(dia_id))
        dia_labels = video_labels[dia_id]
        split_dir_path = os.path.join(params.videos_dir, 'train_splits')
        if dia_id >= (train_dia_num+1):
            split_dir_path = os.path.join(params.videos_dir, 'dev_splits')
            dia_id = dia_id - (train_dia_num+1)
            if dia_id >= val_dia_num:
                continue

        for _index, _id in enumerate(sen_ids):
            print('utterance id {}'.format(_id))
            video_fname = os.path.join(split_dir_path,'dia{}_utt{}.mp4'.format(dia_id, _id))      
            label = dia_labels[_index]
            raw_video_feature = get_raw_feature(video_fname)
            raw_video_feature = F.interpolate(raw_video_feature.unsqueeze(dim = 0), (params.input_d, params.input_h, params.input_w))[0]


            one_hot_index = np.zeros((7))
            one_hot_index[label] = 1
            one_hot_index = torch.tensor(one_hot_index,dtype=torch.float64)
            print('save utterance data to pickle file...')
            pickle.dump([raw_video_feature,one_hot_index],open(os.path.join(params.stored_path,'dia_{}_{}.pkl'.format(dia_id, _id)),'wb'))
            print('Done.')
            
def generate_visual_rep(model, params):
    print('generate visual representation...')
    data = pickle.load(open(params.file_path,'rb'))
    video_ids, video_speakers, video_labels_7,video_text, video_audio, video_sentence, video_act, train_vid, test_vid, video_labels_3 = data
    train_dia_num = 1038
    val_dia_num = len(train_vid) - train_dia_num 
    video_visual = {}
    for dia_id, sen_ids in video_ids.items():
        
        dia_visual = []
        split_dir_path = os.path.join(params.videos_dir, 'train_splits')
        if dia_id >= (train_dia_num+1):
            split_dir_path = os.path.join(params.videos_dir, 'dev_splits')
            dia_id = dia_id - (train_dia_num+1)
            if dia_id >= val_dia_num:
                split_dir_path = os.path.join(params.videos_dir, 'test_splits')
                dia_id = dia_id - val_dia_num
        for _index, _id in enumerate(sen_ids):
            video_fname = os.path.join(split_dir_path,'dia{}_utt{}.mp4'.format(dia_id, _id))      
            raw_video_feature = get_raw_feature(video_fname)
            raw_video_feature = F.interpolate(raw_video_feature.unsqueeze(dim = 0), (params.input_d, params.input_h, params.input_w))
            video_rep = model.get_representation(raw_video_feature.to(params.device))[0].detach().cpu().numpy()
            dia_visual.append(video_rep)
        video_visual[dia_id] = dia_visual
    data = video_ids, video_speakers, video_labels_7,video_text, video_audio, video_visual, video_sentence, video_act, train_vid, test_vid, video_labels_3
    pickle.dump(data,open(params.output_path,'wb'))
            
def set_seed(params):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    random.seed(params.seed)
    os.environ['PYTHONHASHSEED'] = str(params.seed)
    np.random.seed(params.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(params.seed)
        torch.cuda.manual_seed_all(params.seed)
    else:
        torch.manual_seed(params.seed)

def run(params):
    train(params)
    print('loading the pretraining model...')
    model = torch.load('dummy_files/best_model.pt')
    generate_visual_rep(model, params)
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='running experiments on multimodal datasets.')
    parser.add_argument('-config', action = 'store', dest = 'config_file', help = 'please enter configuration file.',default = 'config/pretrain_visual.ini')
    args = parser.parse_args()
    params = Params()
    params.parse_config(args.config_file) 
    params.config_file = args.config_file
    mode = 'run'
    if 'mode' in params.__dict__:
        mode = params.mode
    set_seed(params)
    params.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if mode == 'extract_raw_feature':
        extract_raw_features(params)
    elif mode == 'run':
        run(params)
   
    


    

            
            