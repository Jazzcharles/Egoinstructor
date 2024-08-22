
import csv
import glob
import json
import numpy as np
import os.path as osp
import pickle
import random

import decord
import pandas as pd
import torch
import cv2
import io,os
from copy import deepcopy

from nltk.tokenize import word_tokenize

### 1: inter,  2: intra 
class EGOMCQDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, is_training=False, tokenizer=None):
        ### common setups ###
        self.root = cfg.root
        self.metadata = cfg.metadata
        self.clip_length = cfg.clip_length
        self.clear_narration = cfg.clear_narration

        ### maybe customized ###
        self.is_training = is_training
        self.tokenizer = tokenizer
        with open(self.metadata, 'r') as f:
            self.samples = json.load(f)

        self.load_from = cfg.load_from
        if self.load_from == 'ceph':
            from petrel_client.client import Client
            self.client = Client()
            
    def __len__(self):
        return len(self.samples)
    
    def load_frame_feature(self, vid_path, start_second, end_second):
        filename = os.path.join(self.root, vid_path + '.pth.tar')
        if '.mp4' in filename:
            filename = filename.replace('.mp4','')
        
        if self.load_from == 'ceph':
            meta = self.client.get(filename)
            metabytes = io.BytesIO(meta)
            frame_feat = torch.load(metabytes, map_location='cpu')
        elif self.load_from == 'dir':
            frame_feat = torch.load(filename,  map_location='cpu')
        else:
            raise NotImplementedError
        
        total_seconds = len(frame_feat)
        #assert start_second <= end_second
        duration = end_second - start_second

        ### otherwise, random / center select seconds ###
        seg_size = float(end_second - start_second - 1) / self.clip_length
        seq = []
        for i in range(self.clip_length):
            start = int(np.round(seg_size * i) + start_second)
            end = int(np.round(seg_size * (i + 1)) + start_second)
            
            ### added here to avoid out-of-boundary of frame_id, as np.random.randint ###
            start = min(start, total_seconds - 1)
            end = min(end, total_seconds)

            if self.is_training:
                second_id = np.random.randint(low=start, high=(end + 1))
            else:
                second_id = (start + end) // 2
            # print(start_frame, end_frame, start, end, end_frame, frame_id)
            seq.append(second_id)
        
        seq = np.array(seq)
        return frame_feat[seq], seq
        
    def get_raw_feature(self, i):
        itemMCQ = self.samples[str(i)]
        answerIndex = itemMCQ['answer']
        textQuery = itemMCQ['query']['clip_text']
        sampleOptions = itemMCQ['choices']
        frames_options = []
        narration_options = []
        for option_id in range(len(sampleOptions)):
            option = sampleOptions[str(option_id)]
            frames, second_ids = self.load_frame_feature(option['video_uid'], float(option['clip_start']), float(option['clip_end']))
            frames_options.append(frames)
            narration_options.append(option['clip_text'])
        
        if self.clear_narration:
            textQuery = self.clean_narration(textQuery)
            narration_options = [self.clean_narration(x) for x in narration_options]
        return textQuery, frames_options, narration_options, answerIndex, itemMCQ['types'], second_ids

    def narration_filter(self, x):
        if x in ['#' , 'c' , 'cc', 'o' , 'x', 'y', 'b', 'p', 's', 'r', 'g', 'n', 'z', 'v', 'k']:
            return ''
        return x

    def clean_narration(self, narration):        
        ### clear ego4d data ###
        alltext = word_tokenize(narration.lower())
        # print(alltext)
        filtered_text = [self.narration_filter(x) for x in alltext]
        filtered_text = [x for x in filtered_text if len(x)]
        narration = ' '.join(filtered_text)
        return narration
    
    def __getitem__(self, i):
        ### for record info only ###
        # try:
        textQuery, frames_options, narration_options, answerIndex, q_type, second_ids = self.get_raw_feature(i)
        frames = frames_options

        raw_textQuery = textQuery
        raw_narration_options = narration_options

        textQuery = self.tokenizer(textQuery) if self.tokenizer is not None else None
        narration_options = self.tokenizer(narration_options) if self.tokenizer is not None else None
        
        if isinstance(textQuery, tuple):
            textQuery, textQuery_mask = textQuery
            narration_options, narration_options_mask = narration_options
        else:
            textQuery_mask = torch.zeros_like(textQuery).long()
            textQuery_mask[:torch.where(textQuery == 49407)[0] + 1] = 1
            
            narration_options_mask = torch.zeros_like(narration_options).long()
            last_token = torch.where(narration_options == 49407)[1] + 1
            for i in range(len(narration_options)):
                narration_options_mask[i, :last_token[i]] = 1
    
        return textQuery, torch.stack(frames, dim=0), narration_options, answerIndex, q_type, textQuery_mask, raw_textQuery, raw_narration_options
        
        # except Exception as e:
        #     return self.__getitem__(0)


class EGOSUMMDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, is_training=False, tokenizer=None, shuffle_mcq=False):
        ### common setups ###
        self.root = cfg.root
        self.metadata = cfg.metadata
        self.clip_length = cfg.clip_length
        self.clear_narration = cfg.clear_narration
        self.ctx_length = cfg.ctx_length
        
        ### maybe customized ###
        self.is_training = is_training
        self.tokenizer = tokenizer
        with open(self.metadata, 'r') as f:
            self.samples = json.load(f)

        self.shuffle_mcq = shuffle_mcq
        self.load_from = cfg.load_from
        if self.load_from:            
            from petrel_client.client import Client
            self.client = Client()
            
    def __len__(self):
        return len(self.samples)
    
    def load_frame_feature(self, vid_path, start_second, end_second):
        filename = os.path.join(self.root, vid_path + '.pth.tar')
        if '.mp4' in filename:
            filename = filename.replace('.mp4','')
        
        if self.load_from == 'ceph':
            meta = self.client.get(filename)
            metabytes = io.BytesIO(meta)
            frame_feat = torch.load(metabytes, map_location='cpu')
        elif self.load_from == 'dir':
            frame_feat = torch.load(filename,  map_location='cpu')
        else:
            raise NotImplementedError
        
        total_seconds = len(frame_feat)

        #assert start_second <= end_second
        duration = end_second - start_second

        ### otherwise, random / center select seconds ###
        seg_size = float(end_second - start_second - 1) / self.clip_length
        seq = []
        for i in range(self.clip_length):
            start = int(np.round(seg_size * i) + start_second)
            end = int(np.round(seg_size * (i + 1)) + start_second)
            
            ### added here to avoid out-of-boundary of frame_id, as np.random.randint ###
            start = min(start, total_seconds - 1)
            end = min(end, total_seconds)

            if self.is_training:
                second_id = np.random.randint(low=start, high=(end + 1))
            else:
                second_id = (start + end) // 2
            # print(start_frame, end_frame, start, end, end_frame, frame_id)
            seq.append(second_id)
        
        seq = np.array(seq)
        return frame_feat[seq], seq
        
    def get_raw_feature(self, i):
        itemMCQ = self.samples[str(i)]
        answerIndex = itemMCQ['answer']
        textQuery = itemMCQ['query']['summary_text']
        sampleOptions = itemMCQ['choices']
        frames_options = []
        narration_options = []
        for option_id in range(len(sampleOptions)):
            option = sampleOptions[str(option_id)]
            frames, second_ids = self.load_frame_feature(option['vid'], float(option['start_sec']), float(option['end_sec']))
            frames_options.append(frames)
            narration_options.append(option['summary_text'])
        
        if self.clear_narration:
            textQuery = self.clean_narration(textQuery)
            narration_options = [self.clean_narration(x) for x in narration_options]
        return textQuery, frames_options, narration_options, answerIndex, itemMCQ['types'], second_ids

    def narration_filter(self, x):
        if x in ['#' , 'c' , 'cc', 'o' , 'x', 'y', 'b', 'p', 's', 'r', 'g', 'n', 'z', 'v', 'k']:
            return ''
        return x

    def clean_narration(self, narration):        
        ### clear ego4d data ###
        alltext = word_tokenize(narration.lower())
        # print(alltext)
        filtered_text = [self.narration_filter(x) for x in alltext]
        filtered_text = [x for x in filtered_text if len(x)]
        narration = ' '.join(filtered_text)
        return narration
    
    def __getitem__(self, i):
        ### for record info only ###
        try:
            textQuery, frames_options, narration_options, answerIndex, q_type, second_ids = self.get_raw_feature(i)   
            frames = frames_options

            raw_textQuery = textQuery
            raw_narration_options = narration_options

            textQuery = self.tokenizer(textQuery, self.ctx_length) if self.tokenizer is not None else None
            narration_options = self.tokenizer(narration_options) if self.tokenizer is not None else None
            
            if isinstance(textQuery, tuple):
                textQuery, textQuery_mask = textQuery
                narration_options, narration_options_mask = narration_options
            else:
                textQuery_mask = torch.zeros_like(textQuery).long()
                textQuery_mask[:torch.where(textQuery == 49407)[0] + 1] = 1
                
                narration_options_mask = torch.zeros_like(narration_options).long()
                last_token = torch.where(narration_options == 49407)[1] + 1
                #print('Len_narration:', len(narration_options), len(last_token), last_token)
                for i in range(len(narration_options)):
                    if i < len(last_token):
                        narration_options_mask[i, :last_token[i]] = 1
            
            return textQuery, torch.stack(frames, dim=0), narration_options, answerIndex, q_type, textQuery_mask, raw_textQuery, raw_narration_options
        
        except Exception as e:
            return self.__getitem__(0)