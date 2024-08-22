
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

def datetime2sec(str):
    hh, mm, ss = str.split(':')
    return int(hh) * 3600 + int(mm) * 60 + float(ss)

class EK100Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, is_training=False, tokenizer=None):
        ### common setups ###
        self.root = cfg.root
        self.metadata = cfg.metadata
        self.clip_length = cfg.clip_length
        
        ### maybe customized ###
        self.is_training = is_training
        self.tokenizer = tokenizer
        self.load_from = cfg.load_from
        if self.load_from == 'ceph':
            from petrel_client.client import Client
            self.client = Client()
        
        self.samples = []
        with open(self.metadata) as f:
            csv_reader = csv.reader(f)
            _ = next(csv_reader)  # skip the header
            for row in csv_reader:
                pid, vid = row[1:3]
                # start_frame, end_frame = int(row[6]), int(row[7])
                # Deprecated: some videos might have fps mismatch issue
                start_timestamp, end_timestamp = datetime2sec(row[4]), datetime2sec(row[5])
                narration = row[8]
                verb, noun = int(row[10]), int(row[12])

                vid_path = '{}.mp4'.format(vid)
                self.samples.append((vid_path, start_timestamp, end_timestamp, narration, verb, noun))
        
        self.metadata_sentence = pd.read_csv(self.metadata[:self.metadata.index('.csv')] + '_sentence.csv')
        if 'train' in self.metadata:
            # self.relevancy_mat = pickle.load(open(osp.join(osp.dirname(self.metadata), 'relevancy', 'caption_relevancy_EPIC_100_retrieval_train.pkl'), 'rb'))
            self.relevancy_mat = pickle.load(open(osp.join(osp.dirname(self.metadata), 'caption_relevancy_EPIC_100_retrieval_train.pkl'), 'rb'))
        elif 'test' in self.metadata:
            # self.relevancy_mat = pickle.load(open(osp.join(osp.dirname(self.metadata), 'relevancy', 'caption_relevancy_EPIC_100_retrieval_test.pkl'), 'rb'))
            self.relevancy_mat = pickle.load(open(osp.join(osp.dirname(self.metadata), 'caption_relevancy_EPIC_100_retrieval_test.pkl'), 'rb'))
        else:
            raise ValueError('{} should contain either "train" or "test"!'.format(self.metadata))
        self.relevancy = .1

    def __len__(self):
        return len(self.samples)
    
    def get_raw_feature(self, i):
        vid_path, start_second, end_second, narration, verb, noun = self.samples[i]
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
            seq.append(second_id)
        
        seq = np.array(seq)
        return frame_feat[seq], seq

    def __getitem__(self, i):
        ### for record info only ###
        vid_path, start_timestamp, end_timestamp, narration, verb, noun = self.samples[i]
        uid = vid_path
        raw_caption = narration
        
        frames, second_ids = self.get_raw_feature(i)
        relevancy = 1
        
        caption = self.tokenizer(narration) if self.tokenizer is not None else None

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
                'relevancy': relevancy,
                'caption': caption,
                'raw_caption': raw_caption,
                'success': True,     
                'start': start_timestamp,
                'end': end_timestamp,   
                'second_ids': second_ids,
        }
        

