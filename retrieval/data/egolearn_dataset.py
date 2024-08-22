
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
from decord import cpu
import cv2
import io,os

class EgoLearnDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, transform=None, is_training=False, tokenizer=None):
        ### common setups ###
        self.ego_root = cfg.ego_root
        self.exo_root = cfg.exo_root
        self.metadata = cfg.metadata
        
        self.clip_length = cfg.clip_length
        self.ctx_length = cfg.ctx_length
        self.chunk_len = -1

        ### maybe customized ###
        self.transform = transform
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
    
    def load_frame_feature(self, root, vid_path, start_second, end_second):
        filename = os.path.join(root, vid_path + '.pth.tar')
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
    
    def get_each_video_feature(self, option, ego_flag):
        if ego_flag == 1:
            frames, second_ids = self.load_frame_feature(self.ego_root, option['video_id'], float(option['start_second']), float(option['end_second']))
        else:
            frames, second_ids = self.load_frame_feature(self.exo_root, option['video_id'], float(option['start_second']), float(option['end_second']))
        return frames, second_ids

    def get_raw_feature_v2v(self, i):
        itemMCQ = self.samples[str(i)]
        answerIndex = itemMCQ['answer']
        videoQuery = itemMCQ['query']
        textQuery = videoQuery['text']
        types = itemMCQ['types']               
        
        frameQuery, second_is = self.get_each_video_feature(videoQuery, 1 if types == 1 else 0)
        
        frames_options = []
        narration_options = []
        sampleOptions = itemMCQ['choices']
        for option_id in range(len(sampleOptions)):
            option = sampleOptions[str(option_id)]
            frames, second_ids = self.get_each_video_feature(option, 0 if types == 1 else 1)
            
            frames_options.append(frames)
            narration_options.append(option['text'])
        
        return frameQuery, textQuery, frames_options, narration_options, answerIndex, types

 
    def __getitem__(self, i):
        frameQuery, textQuery, frames_options, narration_options, answerIndex, q_type = self.get_raw_feature_v2v(i)
        raw_textQuery = textQuery
        raw_narration_options = narration_options
        
        return frameQuery, torch.stack(frames_options, dim=0), answerIndex, q_type
    
    