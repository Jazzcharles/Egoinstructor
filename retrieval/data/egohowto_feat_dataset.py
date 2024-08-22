
import csv
import glob
import json
import numpy as np
import os.path as osp
import pickle
import random

import pandas as pd
import torch
import cv2
import io, os

from nltk.tokenize import word_tokenize

class EgoHowToFeatDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, tokenizer, is_training=True):
        self.cfg = cfg
        self.is_training = is_training
        self.tokenizer = tokenizer
        ### for data loading ###
        self.dataset = cfg.dataset
        self.ego_root = cfg.root
        self.ego_metadata = cfg.metadata
        self.howto_root = cfg.howto_root
        self.howto_metadata = cfg.howto_metadata

        self.clip_length = cfg.clip_length
        self.ctx_length = cfg.ctx_length

        self.param_dict = {
            'root': { 0: self.ego_root,  1: self.howto_root},
        }    
        
        assert self.dataset in ['ego4d_feat', 'howto100_feat', 'ego4d_howto100_feat']
        self.ego_samples = json.load(open(self.ego_metadata))
        self.howto_samples = json.load(open(self.howto_metadata))

        self.ego4d_number = len(self.ego_samples)
        ### merge two datasets ###
        self.samples = {}
        if self.dataset == 'ego4d_feat':
            # self.samples.extend(self.ego_samples)
            self.samples = {0: self.ego_samples}
        elif self.dataset == 'howto100_feat':
            # self.samples.extend(self.howto_samples)
            self.samples = {1: self.howto_samples}
        elif self.dataset == 'ego4d_howto100_feat':
            self.samples = {
                0: self.ego_samples,
                1: self.howto_samples,
            }
                    
        self.clear_narration = cfg.clear_narration
        self.expand_period = cfg.expand_period
        self.noun_dim = 582  # num of nouns of ego4d taxonomy dictionary
        self.verb_dim = 118  # num of verbs of ego4d taxonomy dictionary
        self.pair_num = cfg.pair_num 
        self.load_from = cfg.load_from
        if self.load_from == 'ceph':
            from petrel_client.client import Client
            self.client = Client()
               
        print('Done init dataset')

    def __len__(self):
        ego_len = len(self.samples[0]) if 0 in self.samples else 0
        exo_len = len(self.samples[1]) if 1 in self.samples else 0
        return ego_len + exo_len

    def narration_filter(self, x):
        if x in ['#' , 'c' , 'cc', 'o' , 'x', 'y', 'b', 'p', 's', 'r', 'g', 'n', 'z', 'v', 'k']:
            return ''
        return x

    def clean_narration(self, egoexo_flag, narration):        
        ### clear ego4d data ###
        if egoexo_flag == 0:
            alltext = word_tokenize(narration.lower())
            # print(alltext)
            filtered_text = [self.narration_filter(x) for x in alltext]
            filtered_text = [x for x in filtered_text if len(x)]
            narration = ' '.join(filtered_text)
        else:
            ### clear howto data ###
            narration = 'I am doing something.' if not isinstance(narration, str) or len(narration) == 0 else narration
        return narration
                
    def get_random_egoexo_sample_id(self, egoexo_flag):
        if egoexo_flag == 0:
            sample_id = np.random.randint(0, self.ego4d_number)
        else:
            sample_id = np.random.randint(self.ego4d_number, len(self.samples))
        return sample_id
    
    def load_multiple_pair_metadata(self, metadata, egoexo_flag):
        pair_idx = metadata['nv_index']
        selected_idx = np.random.choice(pair_idx, self.pair_num, replace=False)
        ### debug only ####
        # selected_idx = [0] * self.pair_num

        data_list = [self.load_metadata(x, egoexo_flag) for x in selected_idx]
        return data_list

    def load_metadata(self, id_offset, egoexo_flag):
        data = self.samples[egoexo_flag][str(id_offset)]
        data['uid'] = data['uid'] if 'uid' in data else data['vid']
        if self.clear_narration:
            data['text'] = self.clean_narration(egoexo_flag, data['text'])

        return data
    
    def load_video_feature(self, vid, start_second, end_second, egoexo_flag):
        filename = os.path.join(self.param_dict['root'][egoexo_flag], vid + '.pth.tar')
        if self.load_from == 'ceph':
            meta = self.client.get(filename)
            metabytes = io.BytesIO(meta)
            frame_feat = torch.load(metabytes, map_location='cpu')
        elif self.load_from == 'dir':
            frame_feat = torch.load(filename, map_location='cpu')
        else:
            raise NotImplementedError
            
        total_seconds = len(frame_feat)
        ### sanity check ##
        end_second = max(end_second, start_second + 1)
        duration = end_second - start_second 

        ### expand howto data duration ###
        if egoexo_flag == 1 and (end_second - start_second <= 1):
            start_second = max(0, start_second - self.expand_period // 2)
            end_second = min(end_second + self.expand_period // 2, total_seconds)

        ### otherwise, random / center select seconds ###
        seg_size = float(end_second - start_second - 1) / self.clip_length
        seq = []
        for i in range(self.clip_length):
            start = int(np.round(seg_size * i) + start_second)
            end = int(np.round(seg_size * (i + 1)) + start_second)
            
            ### added here to avoid out-of-boundary of frame_id, as np.random.randint ###
            start = min(start, total_seconds)
            end = max(start, min(end, total_seconds))

            if self.is_training:
                second_id = np.random.randint(low=start, high=(end + 1))
            else:
                second_id = (start + end) // 2

            # print(start_frame, end_frame, start, end, end_frame, frame_id)
            seq.append(second_id)
        
        seq = np.array(seq)
        return frame_feat[seq], seq

    def process_nounverb(self, noun_idx, verb_idx):
        noun_vec = torch.zeros(self.noun_dim)
        verb_vec = torch.zeros(self.verb_dim)
        noun_vec[noun_idx] = 1
        verb_vec[verb_idx] = 1
        return noun_vec, verb_vec
    
    def check_valid_pair(self, noun1, verb1, noun2, verb2):
        nounset1, verbset1 = set(noun1), set(verb1)
        nounset2, verbset2 = set(noun2), set(verb2)
        inter_noun = nounset1 & nounset2
        inter_verb = verbset1 & verbset2
        ### approve either of them ###
        # return len(inter_noun) > 0 or len(inter_verb) > 0
        
        ### approve both of them ###
        return len(inter_noun) and len(inter_verb)
    
    def get_noun(self,metadata):
        return metadata['noun'] if 'noun' in metadata else []

    def get_verb(self,metadata):
        return metadata['verb'] if 'verb' in metadata else []

    def __getitem__(self, i):
        ### set an indicator, 0 for ego4d, 1 for howto100
        try:
            if self.dataset == 'ego4d_howto100_feat':
                if i < self.ego4d_number:
                    egoexo_flag = 0
                    id_offset = i
                else:
                    egoexo_flag = 1
                    id_offset = i - self.ego4d_number
            elif self.dataset == 'ego4d_feat':
                egoexo_flag = 0
                id_offset = i
            elif self.dataset == 'howto100_feat':
                egoexo_flag = 1
                id_offset = i

            ret_info = {}

            ### load current data ###
            metadata = self.load_metadata(id_offset, egoexo_flag)
            vid, uid, start_second, end_second, narration = metadata['vid'], metadata['uid'], metadata['start_second'], metadata['end_second'], metadata['text']
            ### load refined text (if possible) ###
            if egoexo_flag == 1 and "refined_text" in metadata:
                narration = metadata["refined_text"]                
            
            frames_feature, second_ids = self.load_video_feature(vid, start_second, end_second, egoexo_flag)
            
            ### get current noun/verb embedding ###
            noun = self.get_noun(metadata)
            verb = self.get_verb(metadata)            
            noun_vec, verb_vec = self.process_nounverb(noun, verb)
            ret_info['noun_vec'] = noun_vec
            ret_info['verb_vec'] = verb_vec    
                    
            pair_frames = []
            pair_text = []   

            ### get pair video and text ###
            if egoexo_flag == 1 or 'nv_index' not in metadata:
                ### for exo-only pairs and no paired data, add themselves as pairs ###
                pair_frames.append(frames_feature)
                pair_text.append(narration)
                pair_info = [metadata]
                
            else:
                ### align ego -> exo pairs only performs better than both directions ###
                pair_egoexo_flag = egoexo_flag ^ 1
                pair_info = self.load_multiple_pair_metadata(metadata, pair_egoexo_flag) 
                for each_pair in pair_info:
                    # pair_noun = self.get_noun(pair_info)
                    # pair_verb = self.get_verb(pair_info)
                    pair_noun = self.get_noun(each_pair)
                    pair_verb = self.get_verb(each_pair)

                    ### sanity check,  whether the paired data is valid (has same noun / verb) ###
                    if self.check_valid_pair(noun, verb, pair_noun, pair_verb):                    
                        ### if valid, add the pair ###
                        curr_pair_frame, curr_second_ids = self.load_video_feature(each_pair['vid'], each_pair['start_second'], each_pair['end_second'], pair_egoexo_flag)
                        pair_frames.append(curr_pair_frame)
                        pair_text.append(each_pair['text'])
                    else:
                        ### if not valid, just add the current item itself ###
                        pair_frames.append(frames_feature)
                        pair_text.append(narration)

            pair_frames = torch.stack(pair_frames, dim=0)
            pair_raw_caption = pair_text
            
            ### concat item and its pair ###
            frames_feature = torch.cat([frames_feature.unsqueeze(0), pair_frames], dim = 0)
            narration = [narration] + pair_text    
            if egoexo_flag == 1:
                ### for exo, only add its own nouns and verbs ###
                all_nouns = [noun_vec] * (1 + self.pair_num)
                all_verbs = [verb_vec] * (1 + self.pair_num)
            else:
                ### for ego, add paired nouns and verbs ###
                all_vecs = [self.process_nounverb(self.get_noun(x), self.get_verb(x)) for x in pair_info]
                all_nouns = [noun_vec] + [x[0] for x in all_vecs]
                all_verbs = [verb_vec] + [x[1] for x in all_vecs]

            ret_info['noun_vec'] = torch.stack(all_nouns, dim=0)
            ret_info['verb_vec'] = torch.stack(all_verbs, dim=0)

            if self.tokenizer is not None:
                caption = self.tokenizer(narration, self.ctx_length)
                
            ret_info['uid'] = uid
            ret_info['vid'] = vid
            ret_info['video'] = frames_feature
            ret_info['text'] = caption
            ret_info['raw_caption'] = narration
            ret_info['second_ids'] = second_ids
            return ret_info

        except Exception as e:
            print(f'Error loading {vid} with {e}')
            idx = np.random.randint(0, 10000)
            return self.__getitem__(idx)


if __name__ == '__main__':
    ego_rootdir = 'myphdd:s3://my_ego4d/internvideo_MM_L14_features/'
    exo_rootdir = 'myphdd:s3://HT100M/internvideo_MM_L14/'

    ego_meta = '/mnt/petrelfs/xujilan/data/ego4d/generated/ego4d_cooking_train_clips.json'
    exo_meta = '/mnt/petrelfs/xujilan/data/howto100/generated/howto100_equal_ego4d_nounverb.json'
    
    cfg = None
    tokenizer = None
    dataset = EgoHowToFeatDataset(
        cfg=cfg,
        dataset='howto100',
        ego_root=ego_rootdir,
        ego_metadata=ego_meta,
        howto_root=exo_rootdir,
        howto_metadata=exo_meta,
        tokenizer=tokenizer,
    )
    for i in range(10):
        t = dataset.__getitem__(0)
        print(i, t['video'].shape, t['raw_caption'])
    st()
