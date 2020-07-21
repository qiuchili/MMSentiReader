# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import pickle
import os
import string
import re
import pandas as pd

def add_dialogue_act(dataset_name, act_annotation_file, input_pickle_file, output_pickle_file):
    if dataset_name == 'IEMOCAP':
        add_IEMOCAP_dialogue_act(act_annotation_file, input_pickle_file, output_pickle_file)
    elif dataset_name == 'MELD':
        add_MELD_dialogue_act(act_annotation_file, input_pickle_file, output_pickle_file)
        
def add_MELD_dialogue_act(act_annotation_file, input_pickle_file, output_pickle_file):

    video_dialogue_acts = {}
    data = pickle.load(open(input_pickle_file,'rb'), encoding='latin1')
    video_ids, video_speakers, video_labels_7,video_text, video_audio, video_sentence, train_vid, test_vid, video_labels_3 = data
    table = pd.read_csv(act_annotation_file)
    utt_count = 0
    for i in video_ids:
        _dia_acts_i = []
        _dia_ids = video_ids[i]
        for _utt_id in _dia_ids:
            
            eda1 = table['eda1'][utt_count]
            eda2 = table['eda2'][utt_count]
            eda3 = table['eda3'][utt_count]
            eda4 = table['eda4'][utt_count]
            eda5 = table['eda5'][utt_count]
            eda = table['EDA'][utt_count]
            _dia_acts_i.append([eda1, eda2,eda3,eda4,eda5,eda])
            
            _utt_id = _utt_id[-3:]
            print(table['utt_id'][utt_count], _utt_id)
#            if not table['utt_id'][utt_count] == _utt_id:
#                break
            utt_count = utt_count +1
            
        video_dialogue_acts[i] = _dia_acts_i
    data = video_ids, video_speakers, video_labels_7,video_text, video_audio, video_sentence, video_dialogue_acts, train_vid, test_vid, video_labels_3 
    pickle.dump(data, open(output_pickle_file,'wb'))
        
def add_IEMOCAP_dialogue_act(act_annotation_file, input_pickle_file, output_pickle_file):
    
    video_dialogue_acts = {}
    data = pickle.load(open(input_pickle_file,'rb'), encoding='latin1')
    video_ids, video_speakers, video_labels, video_text, video_audio, video_visual, video_sentence, train_vid, test_vid = data
    table = pd.read_csv(act_annotation_file)

    for i in video_ids:
        _dia_acts_i = []
        _dia_ids = video_ids[i]
        for _id in _dia_ids:
            
            _utt_id = int(_id[-3:])
            _spk_id = _id[:-3]
            
            match = (table['speaker'] == "b'{}'".format(_spk_id)) & (table['utt_id'] == _utt_id)
            
            
            eda1 = table['eda1'][match].values[0]
            eda2 = table['eda2'][match].values[0]
            eda3 = table['eda3'][match].values[0]
            eda4 = table['eda4'][match].values[0]
            eda5 = table['eda5'][match].values[0]
            eda = table['EDA'][match].values[0]
            if not len(table['eda1'][match]) == 1:
                print('wrong!')
                break
            _dia_acts_i.append([eda1, eda2,eda3,eda4,eda5,eda])
            
        video_dialogue_acts[i] = _dia_acts_i
    data = video_ids, video_speakers, video_labels, video_text, video_audio, video_visual, video_sentence, video_dialogue_acts, train_vid, test_vid
    pickle.dump(data, open(output_pickle_file,'wb'))
        
def read_conv_emotion_data(in_path, out_path, dataset_name, max_utt_len = 20):
    if dataset_name == 'AVEC':
        read_AVEC_pickle_data(in_path,out_path, max_utt_len)
    elif dataset_name == 'IEMOCAP':
        read_IEMOCAP_pickle_data(in_path,out_path,max_utt_len)
    elif dataset_name == 'MELD':
        read_MELD_pickle_data(in_path,out_path,max_utt_len)
        
def read_AVEC_pickle_data(input_folder,output_folder, max_utt_len = 20):
#    input_file = 'D:/qiuchi/python/conv-emotion/DialogueRNN/AVEC_features/AVEC_features_1.pkl'
#    input_folder = 'D:/qiuchi/python/conv-emotion/DialogueRNN/AVEC_features/'
#    output_folder =  'D:/qiuchi/data/multimodal/MELD/AVEC_features'
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    emotion_dimensions = ['Valence','Arousal','Expectancy','Power']
    for i, _dim in enumerate(emotion_dimensions):
        input_file = os.path.join(input_folder, 'AVEC_features_{}.pkl'.format(i+1))
        output_file = os.path.join(output_folder, 'avec_data_{}.pkl'.format(_dim))
        data = pickle.load(open(input_file,'rb'), encoding='latin1')
        ratio = 0.2
        video_ids, video_speakers, video_labels, video_text, video_audio, video_visual, video_sentence,train_vid, test_vid = data
        train_vid = list(train_vid)
        valid_vid = train_vid[-int(len(train_vid)*ratio):]
        train_vid = train_vid[:-int(len(train_vid)*ratio)]
        train_text = []
        train_acoustic = []
        train_visual = []
        train_words = []
        train_speaker_ids = []
        train_emotion = []
        
        test_text = []
        test_acoustic = []
        test_words = []
        test_visual = []
        test_speaker_ids = []
        test_emotion = []
        
        dev_text = []
        dev_acoustic = []
        dev_visual = []
        dev_words = []
        dev_speaker_ids = []
        dev_emotion = []
        
        speakers = ['operator','user']
        speaker_num = len(speakers)
        
        for _id in video_ids:
            emotion = video_labels[_id]
            
            dialogue_text = []
            for s in video_sentence[_id]:
                out_str = re.sub('[{}]'.format(re.escape(string.punctuation)), ' ', s.decode('utf-8'))
                out_str = out_str.lower().split()
#                out_str = [w for w in out_str if w not in stopwords.words('english')]
                out_str = pad_seq(out_str,max_utt_len)
                dialogue_text.append(out_str)
#        tokens = [w for w in tokens if w not in 
            
            if _id in train_vid:
                train_text.append(video_text[_id])
                train_acoustic.append(video_audio[_id])
                train_visual.append(video_visual[_id])
                train_words.append(dialogue_text)
                train_speaker_ids.append([speakers.index(s) for s in video_speakers[_id]])
                train_emotion.append(emotion)
            elif _id in test_vid:
                test_text.append(video_text[_id])
                test_acoustic.append(video_audio[_id])
                test_visual.append(video_visual[_id])
                test_words.append(dialogue_text)
                test_speaker_ids.append([speakers.index(s) for s in video_speakers[_id]])
                test_emotion.append(emotion)
            elif _id in valid_vid:
                dev_text.append(video_text[_id])
                dev_acoustic.append(video_audio[_id])
                dev_visual.append(video_visual[_id])
                dev_words.append(dialogue_text)
                dev_speaker_ids.append([speakers.index(s) for s in video_speakers[_id]])
                dev_emotion.append(emotion)
        train_data = {'text':train_words,'audio': train_acoustic, 'vision':train_visual, 'language': train_text,'emotion': train_emotion,'speaker_ids':train_speaker_ids}
        test_data = {'text':test_words,'audio': test_acoustic, 'vision':test_visual, 'language': test_text, 'emotion': test_emotion,'speaker_ids':test_speaker_ids}
        dev_data = {'text':dev_words,'audio': dev_acoustic, 'vision':dev_visual, 'language': dev_text,'emotion': dev_emotion,'speaker_ids':dev_speaker_ids}
        data = {'train':train_data, 'test':test_data, 'valid':dev_data,'speaker_num':speaker_num}
        pickle.dump(data,open(output_file,'wb'))

    
def read_IEMOCAP_pickle_data(input_file, output_file, max_utt_len = 20):
#    input_file = 'D:/qiuchi/python/conv-emotion/DialogueRNN/IEMOCAP_features/IEMOCAP_features_raw.pkl'
#    output_file = 'D:/qiuchi/data/multimodal/MELD/iemocap_data_6emo.pkl'

    #open('D:/qiuchi/python/conv-emotion/DialogueRNN/IEMOCAP_features/IEMOCAP_features_raw.pkl','rb'), encoding='latin1'
    data = pickle.load(open(input_file,'rb'), encoding='latin1')
    from mmsdk.mmdatasdk.dataset.standard_datasets.IEMOCAP import standard_folds
    video_ids, video_speakers, video_labels, video_text, video_audio, video_visual, video_sentence, video_act, train_vid, test_vid = data
    train_text = []
    train_acoustic = []
    train_visual = []
    train_words = []
    train_speaker_ids = []
    train_emotion = []
    train_act = []
    
    test_text = []
    test_acoustic = []
    test_words = []
    test_visual = []
    test_speaker_ids = []
    test_emotion = []
    test_act = []
    
    dev_text = []
    dev_acoustic = []
    dev_visual = []
    dev_words = []
    dev_speaker_ids = []
    dev_emotion = []
    dev_act = []
    
    emotions = ['happy','sad','neutral','angry','excited','frustrated']
    speakers = ['F','M']
    speaker_num = len(speakers)
    
    for _id in video_ids:
        emotion = []
        for _i in video_labels[_id]:
            one_hot_indexes = np.zeros((6))
            one_hot_indexes[_i] = 1
            emotion.append(one_hot_indexes)
        dialogue_text = []
        for s in video_sentence[_id]:
            out_str = re.sub('[{}]'.format(re.escape(string.punctuation)), ' ', s)
            out_str = out_str.lower().split()
#            out_str = [w for w in out_str if w not in stopwords.words('english')]
            out_str = pad_seq(out_str,max_utt_len)
            dialogue_text.append(out_str)

        if _id in standard_folds.standard_train_fold:
            train_text.append(video_text[_id])
            train_acoustic.append(video_audio[_id])
            train_visual.append(video_visual[_id])
            train_words.append(dialogue_text)
            train_speaker_ids.append([speakers.index(s) for s in video_speakers[_id]])
            train_emotion.append(emotion)
            train_act.append(video_act[_id])
            
        elif _id in standard_folds.standard_test_fold:
            test_text.append(video_text[_id])
            test_acoustic.append(video_audio[_id])
            test_visual.append(video_visual[_id])
            test_words.append(dialogue_text)
            test_speaker_ids.append([speakers.index(s) for s in video_speakers[_id]])
            test_emotion.append(emotion)
            test_act.append(video_act[_id])

        else:
            dev_text.append(video_text[_id])
            dev_acoustic.append(video_audio[_id])
            dev_visual.append(video_visual[_id])
            dev_words.append(dialogue_text)
            dev_speaker_ids.append([speakers.index(s) for s in video_speakers[_id]])
            dev_emotion.append(emotion)
            dev_act.append(video_act[_id])

            
    train_data = {'text':train_words,'audio': train_acoustic, 'vision':train_visual, 'language': train_text, 'emotion': train_emotion, 'act': train_act,'speaker_ids':train_speaker_ids}
    test_data = {'text':test_words,'audio': test_acoustic, 'vision':test_visual,'language': test_text, 'emotion': test_emotion, 'act': test_act, 'speaker_ids':test_speaker_ids}
    dev_data = {'text':dev_words,'audio': dev_acoustic, 'vision':dev_visual, 'language': dev_text, 'emotion': dev_emotion, 'act': dev_act, 'speaker_ids':dev_speaker_ids}
    data = {'train':train_data, 'test':test_data, 'valid':dev_data,'speaker_num':speaker_num,'emotion_dic':emotions}
    pickle.dump(data,open(output_file,'wb'))

def read_MELD_pickle_data(input_file, output_file, max_utt_len = 20):
    data = pickle.load(open(input_file,'rb'), encoding='latin1')
    video_ids, video_speakers, video_labels_7,video_text, video_audio, video_visual, video_sentence, video_act, train_vid, test_vid, video_labels_3 = data
    train_text = []
    train_acoustic = []
    train_visual = []
    train_words = []
    train_speaker_ids = []
    train_emotion = []
    train_sentiment = []
    train_act = []
    
    test_text = []
    test_acoustic = []
    test_visual = []
    test_words = []
    test_speaker_ids = []
    test_emotion = []
    test_sentiment = []
    test_act = []
    
    dev_text = []
    dev_acoustic = []
    dev_visual = []
    dev_words = []
    dev_speaker_ids = []
    dev_emotion = []
    dev_sentiment = []
    dev_act = []
    
    emotions = ['neutral','surprise','fear','sadness','joy','disgust','anger']
    speaker_num = len(video_speakers[0][0])
    
    for _id in video_ids:
        emotion = []
        for _i in video_labels_7[_id]:
            one_hot_indexes = np.zeros((7))
            one_hot_indexes[_i] = 1
            emotion.append(one_hot_indexes)
            
        dialogue_text = []
        for s in video_sentence[_id]:
            s = s.replace('x92',"'")
            out_str = re.sub('[{}]'.format(re.escape(string.punctuation)), ' ', s)
            out_str = out_str.lower().split()
#            out_str = [w for w in out_str if w not in stopwords.words('english')]
            out_str = pad_seq(out_str,max_utt_len)
            dialogue_text.append(out_str)
        
        #0 -->neutral-->0
        #1 -->positive-->1
        #2-->negative-->-1
        sentiment = [-1 if _i==2 else _i for _i in video_labels_3[_id]]
        sentiment = np.asarray([sentiment])    
        
        if _id in train_vid and _id<=1038:
            train_text.append(video_text[_id])
            train_acoustic.append(video_audio[_id].tolist())
            train_visual.append(video_visual[_id])
            train_words.append(dialogue_text)
            train_speaker_ids.append([s.index(1) for s in video_speakers[_id]])
            train_emotion.append(emotion)
            train_sentiment.append(sentiment)
            train_act.append(video_act[_id])
        elif _id in test_vid:
            test_text.append(video_text[_id])
            test_acoustic.append(video_audio[_id].tolist())
            test_visual.append(video_visual[_id])
            test_words.append(dialogue_text)
            test_speaker_ids.append([s.index(1) for s in video_speakers[_id]])
            test_emotion.append(emotion)
            test_sentiment.append(sentiment)
            test_act.append(video_act[_id])
        else:
            dev_text.append(video_text[_id])
            dev_acoustic.append(video_audio[_id].tolist())
            dev_visual.append(video_visual[_id])
            dev_words.append(dialogue_text)
            dev_speaker_ids.append([s.index(1) for s in video_speakers[_id]])
            dev_emotion.append(emotion)
            dev_sentiment.append(sentiment)
            dev_act.append(video_act[_id])

    train_data = {'text':train_words,'audio': train_acoustic, 'vision':train_visual,'language': train_text, 'sentiment': train_sentiment,'emotion': train_emotion,'act': train_act, 'speaker_ids':train_speaker_ids}
    test_data = {'text':test_words,'audio': test_acoustic, 'vision':test_visual,'language': test_text, 'sentiment': test_sentiment,'emotion': test_emotion, 'act': test_act, 'speaker_ids':test_speaker_ids}
    dev_data = {'text':dev_words,'audio': dev_acoustic, 'vision':dev_visual,'language': dev_text, 'sentiment': dev_sentiment,'emotion': dev_emotion, 'act': dev_act, 'speaker_ids':dev_speaker_ids}
    data = {'train':train_data, 'test':test_data, 'valid':dev_data,'speaker_num':speaker_num,'emotion_dic':emotions}
    pickle.dump(data,open(output_file,'wb'))
    
def pad_seq(f, max_seq_len, pad_type = 'post', pad_token = 'UNK'):
    output = None
    if type(f) == np.ndarray:
        if len(f) > max_seq_len:
            output = f[:max_seq_len]
        else:
            zeros_array =  np.zeros((max_seq_len-len(f),*f.shape[1:]))
            if pad_type == 'pre':
                output = np.concatenate([zeros_array,f])
            elif pad_type == 'post':
                output = np.concatenate([f,zeros_array])
    else:
        str_list = f
        str_list = [pad_token if token =='sp' else token for token in str_list]
        if len(str_list) > max_seq_len:
            output = str_list[:max_seq_len]
        elif pad_type == 'pre':
            output = [pad_token] * (max_seq_len - len(str_list))+ str_list
        elif pad_type == 'post':
            output = str_list + [pad_token] * (max_seq_len - len(str_list))
    return output