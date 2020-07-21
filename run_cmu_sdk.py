# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 11:44:48 2019

@author: qiuchi
"""
import torch
from utils.params import Params
from mmsdk import mmdatasdk
import os
import io
import numpy as np
import h5py
import utils.align_functions as align_functions
import dataset
import argparse
from utils.const import FEATURE_NAME_DIC

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def comp_seq_concat(seq_list, markers):
    
    bio = io.BytesIO()
    fx = h5py.File(bio)
    d = {}
    
    s = set(seq_list[0].keys())
    for seq in seq_list:
        s = s.intersection(set(seq.keys()))
    
    for k in s:
        groups = [seq[k] for seq in seq_list]
        group = fx.create_group(k)
        intervals = [ _g['intervals'].value for _g in groups] 
        concatenated_intervals = np.concatenate(intervals)
        sorted_idx = np.argsort(concatenated_intervals, axis = 0)[:,0]
        original_type = np.dtype('S13')
        features = []
        for i, _g in enumerate(groups):
            feature = _g['features'].value.astype('<U5')
            feature_marked = []
            for f in feature:
                feature_marked.append([f[0]+markers[i]])
            feature_marked = np.asarray(feature_marked)
            features.append(feature_marked)

        features = np.concatenate(features)[sorted_idx,:]
        intervals = concatenated_intervals[sorted_idx,:]
        itv_dataset = fx.create_dataset("intervals_{}".format(k),data = intervals)
        group['intervals'] = itv_dataset
        fea_dataset = fx.create_dataset("features_{}".format(k),data = features.astype(original_type))
        group['features'] = fea_dataset
        
        d[k] = group
    return d


def comp_seq_stack(seq_list):
    
    bio = io.BytesIO()
    fx = h5py.File(bio)
    d = {}
    
    s = set(seq_list[0].keys())
    for seq in seq_list:
        s = s.intersection(set(seq.keys()))
        
    for k in s:
        groups = [seq[k] for seq in seq_list]

        group = fx.create_group(k)
        features = [_g['features'] for _g in groups]
        if all(_f.shape == features[0].shape for _f in features):
            features = np.stack(features,axis=-1)
            itv_dataset = fx.create_dataset("intervals_{}".format(k),data = groups[0]['intervals'])
            group['intervals'] = itv_dataset
            fea_dataset = fx.create_dataset("features_{}".format(k),data = features)
            group['features'] = fea_dataset
            
            d[k] = group

    return d

def download(params):
    print('download {} dataset begins!'.format(params.dataset_name))
    dataset_dic = {"cmumosei":mmdatasdk.cmu_mosei.highlevel,"cmumosi":mmdatasdk.cmu_mosi.highlevel,"pom":mmdatasdk.pom.highlevel}
    label_dic = {"cmumosei":mmdatasdk.cmu_mosei.labels,"cmumosi":mmdatasdk.cmu_mosi.labels,"pom":mmdatasdk.pom.labels}
    raw_dic = {"cmumosei":mmdatasdk.cmu_mosei.raw,"cmumosi":mmdatasdk.cmu_mosi.raw,"pom":mmdatasdk.pom.raw}
    dataset_dir = os.path.join(params.datasets_dir, params.dataset_name) 
        
    dataset = mmdatasdk.mmdataset(dataset_dic[params.dataset_name], dataset_dir+'/')
    dataset.add_computational_sequences(label_dic[params.dataset_name],dataset_dir+'/')
    dataset.add_computational_sequences(raw_dic[params.dataset_name],dataset_dir+'/')

def align(params):
    feature_dic = FEATURE_NAME_DIC[params.dataset_name]
    dataset_dir = os.path.join(params.datasets_dir, params.dataset_name) 

    # Load the features
    
    recipe = {}
    for modality in ['acoustic','textual','visual','visual_2', 'textual_2']:
        if modality in feature_dic:
            feature_name = feature_dic[modality][0]
            feature_file_name = feature_dic[modality][1] + '.csd'
            recipe[feature_name] = os.path.join(dataset_dir,feature_file_name)
        
    dataset = mmdatasdk.mmdataset(recipe)  
    if params.dataset_name == 'iemocap':
        w_1 = dataset.computational_sequences['words'].data
        w_2 = dataset.computational_sequences['words 2'].data
        
        d = comp_seq_concat([w_1, w_2], feature_dic['speaker_markers'])
        dataset.computational_sequences['words'].data = d
        del dataset.computational_sequences['words 2']
    
    # Indentify the alignment feature and the alignment function
    # And align the data
    if params.modality in ['acoustic','textual','visual']:
        feature_name = feature_dic[params.modality][0]
        
        if params.align_function == 'none':
            dataset.align(feature_name)
        else:
            if hasattr(align_functions, params.align_function): 
                align_func = getattr(align_functions, params.align_function)
            else:
                print('Wrong alignment function! The default average function is implemented.')
                align_func = getattr(align_functions, 'avg')
            dataset.align(feature_name,collapse_functions = [align_func])
    
    # Load the labels
    label_recipe = {}
    for l in ['emotion','sentiment','persuasion','emotion_2']:
        if l in feature_dic:
            l_name = feature_dic[l][0]
            l_file_name = feature_dic[l][1] + '.csd'
            label_recipe[l_name] = os.path.join(dataset_dir,l_file_name) 
    dataset.add_computational_sequences(label_recipe, destination = None)
    
    if params.dataset_name == 'iemocap':
        w_1 = dataset.computational_sequences['Emotion Labels'].data
        w_2 = dataset.computational_sequences['Emotion Labels 2'].data
        d = comp_seq_concat([w_1, w_2], feature_dic['speaker_markers'])
        dataset.computational_sequences['Emotion Labels'].data = d
        del dataset.computational_sequences['Emotion Labels 2']
   
    
    # Align the data with the labels
    # If the alignment modality is a feature then no collapse function
    # If the alignment modality is a label then do alignment
    if params.modality in label_recipe and hasattr(align_functions, params.align_function): 
        align_func = getattr(align_functions, params.align_function)
        dataset.align(feature_dic[params.modality][0],collapse_functions = [align_func])
    else:        
        #Pick any label to align, because all labels correspond to the same intervals
        label_name = list(label_recipe.keys())[0]
        dataset.align(label_name)
        
    if params.dataset_name == 'iemocap':
        w_1 = dataset.computational_sequences['OpenFace'].data
        w_2 = dataset.computational_sequences['OpenFace 2'].data
        
        d = comp_seq_stack([w_1, w_2])
        dataset.computational_sequences['OpenFace'].data = d
        del dataset.computational_sequences['OpenFace 2']
        
    # Deploy the aligned features to file
    deploy_files={x:x for x in dataset.computational_sequences.keys()}
    
    if 'align_output_dir' in params.__dict__: 
        output_dir = os.path.join(params.datasets_dir, params.align_output_dir) 
    else:
        align_output_dir = '{}_{}_{}'.format(params.dataset_name,params.modality,params.align_function)
        output_dir = os.path.join(params.datasets_dir, align_output_dir)
    dataset.deploy(output_dir,deploy_files)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='running experiments on multimodal datasets.')
    parser.add_argument('-config', action = 'store', dest = 'config_file', help = 'please enter configuration file.',default = 'config/extract.ini')
    args = parser.parse_args()
    params = Params()
    params.parse_config(args.config_file) 
    params.config_file = args.config_file
    mode = 'extract'
    if 'mode' in params.__dict__:
        mode = params.mode
        
    if mode == 'download':
        download(params)
        
    elif mode == 'align':
        align(params)
        
    elif mode == 'extract':
        reader = dataset.setup(params)
        reader.read_data_from_sdk()

        

