
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
from ipdb import set_trace
import cv2
import io,os

class YouCookDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, is_training=False, tokenizer=None):
        ### common setups ###
        self.root = cfg.root
        self.metadata = cfg.metadata
        self.clip_length = cfg.clip_length
        
        ### maybe customized ###
        self.is_training = is_training
        self.tokenizer = tokenizer
        self.ctx_length = cfg.ctx_length
        
        self.samples = json.load(open(self.metadata))
        self.load_from = cfg.load_from
        if self.load_from == 'ceph':
            from petrel_client.client import Client
            self.client = Client()

    def __len__(self):
        return len(self.samples)
        
    def get_raw_feature(self, i):
        data = self.samples[str(i)]
        vid_path, start_second, end_second, task, narration = data['video_id'], data['start_second'], data['end_second'], data['task'], data['text']
        filename = os.path.join(os.path.join(self.root, str(task)), vid_path + '.pth.tar')
        if '.mp4' in filename:
            filename = filename.replace('.mp4','')
        if '.mkv' in filename:
            filename = filename.replace('.mkv','')
        if '.webm' in filename:
            filename = filename.replace('.webm','')
    
        if self.load_from == 'ceph':
            meta = self.client.get(filename)
            metabytes = io.BytesIO(meta)
            frame_feat = torch.load(metabytes, map_location='cpu')
        elif self.load_from == 'dir':
            frame_feat = torch.load(filename,  map_location='cpu')
        else:
            raise NotImplementedError
        
        total_seconds = len(frame_feat)

        assert start_second <= end_second
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

    def __getitem__(self, i):
        ### for record info only ###
        data = self.samples[str(i)]
        vid, start_second, end_second, task, narration = data['video_id'], data['start_second'], data['end_second'], data['task'], data['text']
        uid = vid

        frames, second_ids = self.get_raw_feature(i)
        
        raw_caption = narration
        caption = self.tokenizer(narration, self.ctx_length) if self.tokenizer is not None else None

        if isinstance(caption, tuple):
            caption, mask = caption
        else:
            mask = torch.zeros_like(caption).long()
            mask[:torch.where(caption == 49407)[0] + 1] = 1
        
        return {
                'video': frames, 
                'text': caption, 
                'uid': uid, 
                'mask': mask, 
                'caption': caption,
                'raw_caption': raw_caption,
                'success': True,        
                'second_ids': second_ids,
                'start': start_second,
                'end': end_second,
        }
        