
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
from decord import cpu
import cv2
import io,os

class CharadesEgoExoDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, is_training=False, tokenizer=None):
        ### common setups ###
        self.ego_root = cfg.ego_root
        self.exo_root = cfg.exo_root
        self.metadata = cfg.metadata
        self.samples = json.load(open(self.metadata))
        self.clip_length = cfg.clip_length

        self.load_from = cfg.load_from
        if self.load_from == 'ceph':
            from petrel_client.client import Client
            self.client = Client()
            
        ### maybe customized ###
        self.is_training = is_training
        self.tokenizer = tokenizer

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
        ### this is for v2v retrieval ###
        metainfo = self.samples[str(i)]
        ego_feat, _ = self.load_frame_feature(
            root = self.ego_root,
            vid_path = metainfo['ego_vid'],
            start_second = metainfo['ego_start_second'],
            end_second = metainfo['ego_end_second'],
        )
        exo_feat, _ = self.load_frame_feature(
            root = self.exo_root,
            vid_path = metainfo['exo_vid'],
            start_second = metainfo['exo_start_second'],
            end_second = metainfo['exo_end_second'],
        )
        return ego_feat, exo_feat