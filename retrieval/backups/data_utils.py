
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
import argparse
from ipdb import set_trace as st

try:
    from petrel_client.client import Client
    client = Client()

    # Disable boto logger
    import logging
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('nose').setLevel(logging.WARNING)
except:
    client = None


def datetime2sec(str):
    hh, mm, ss = str.split(':')
    return int(hh) * 3600 + int(mm) * 60 + float(ss)

def get_vr(video_path):
    # print(video_path)
    # set_trace()
    
    video_bytes = client.get(video_path, enable_stream=True)
    assert video_bytes is not None, "Get video failed from {}".format(video_path)
    video_path = video_bytes
    if isinstance(video_path, bytes):
        video_path = io.BytesIO(video_bytes)
    vreader = decord.VideoReader(video_path, ctx=cpu(0))
    return vreader

def video_loader_by_pt(root, vid, second=None, end_second=None, chunk_len=60, fps=30, sample_stride=4, clip_length=4, is_training=False):
    '''
    args:
        sample_stride: as the pt is obtained with stride > 1, the actual frame indice should be (second * fps // stride)
    '''
    ### changed here to load chunked data ###
    chunk_id = int(second) // chunk_len
    chunk_start = chunk_id * chunk_len
    second_offset = second - chunk_start
    try:    
        vid_path = os.path.join(root, vid, f'{str(chunk_id).zfill(3)}.pt')    
        with io.BytesIO(client.get(vid_path)) as buffer:
            frames = torch.load(buffer, map_location='cpu')    
    except:
        vid_path = os.path.join(root, vid, f'{str(0).zfill(3)}.pt')    
        with io.BytesIO(client.get(os.path.join(root, vid))) as bytes_pt:
            frames = np.load(bytes_pt)
    
    total_frames = frames.shape[0]
    
    start_frame = int(np.round(fps * second_offset))
    total_duration = max(int((end_second - second) * fps), clip_length)
    # end_frame = int(np.ceil(fps * end_timestamp) // sample_stride) if end_timestamp else len(vr) - 1
    frame_ids = get_frame_ids(start_frame // sample_stride, (start_frame + total_duration) // sample_stride, num_segments=clip_length, jitter=is_training)
    
    if max(frame_ids) < total_frames:
        frames = frames[frame_ids].float()
    else:
        ### find the remaining frames in the next chunk ###
        frame_ids_part1 = list(filter(lambda frame_id: frame_id < total_frames, frame_ids))
        frames_part1 = frames[frame_ids_part1]
        
        vid_path2 = os.path.join(root, vid, f'{str(chunk_id + 1).zfill(3)}.pt')    
        with io.BytesIO(client.get(vid_path2)) as buffer2:
            frames2 = torch.load(buffer2, map_location='cpu')
        total_frames2 = frames2.shape[0]    
        
        frame_ids_part2 = list(filter(lambda frame_id: frame_id >= total_frames, frame_ids))
        frame_ids_part2 = [min(frame_id % total_frames, total_frames2 - 1) for frame_id in frame_ids_part2]
        frames_part2 = frames[[frame_ids_part2]]
        frames = torch.tensor(np.concatenate([frames_part1, frames_part2], axis=0)).float()
        
    # print(vid, second, end_second, chunk_id, chunk_start, second_offset, start_frame // sample_stride, (start_frame + total_duration) // sample_stride)
        
    ### bs, 3, 170, 128, manually resize to 224 and normalize ###
    mean = torch.tensor([122.7709393, 116.7460125, 104.09373615000001])
    std = torch.tensor([68.5005327, 66.6321579, 70.32316305])
    
    ### have errors here ###
    #### index 1 is out of bounds for dimension 1 with size 1 
    
    frames[:,0].sub_(mean[0]).div_(std[0])
    frames[:,1].sub_(mean[1]).div_(std[1])
    frames[:,2].sub_(mean[2]).div_(std[2])
    
    frames = torch.nn.functional.interpolate(frames, size=(224, 224), mode='bilinear')
    return frames    
    
    
def video_loader_by_frames_novel(root, vid, start_timestamp=0, end_timestamp=0, clip_length=4, is_training=False):
    ## change calculate start/end frame here
    vr = get_vr(osp.join(root, vid))
    fps = vr.get_avg_fps()
    start_frame = int(np.round(fps * start_timestamp)) if start_timestamp else 0
    end_frame = int(np.ceil(fps * end_timestamp)) if end_timestamp else len(vr) - 1
    frame_ids = get_frame_ids(start_frame, end_frame, num_segments=clip_length, jitter=is_training)
    
    try:
        frames = vr.get_batch(frame_ids).asnumpy()
        frames = [torch.tensor(frame, dtype=torch.float32) for frame in frames]
    except (IndexError, decord.DECORDError) as error:
        print(error)
        print("Erroneous video: ", vid)
        frames = [torch.zeros((240, 320, 3)) for _ in range(len(frame_ids))]
    return torch.stack(frames, dim=0)

def video_loader(root, vid, second=None, end_second=None, chunk_len=300, fps=30, clip_length=32, jitter=False):
    '''
    args:
        root: root directory of the video
        vid: the unique vid of the video, e.g. hello.mp4 
        second: the start second of the clip/video
        end_second: the end second of the clip/video
        chunk_len: whether the (long) video is chunked into several clips (e.g. 300-seconds clip)
        fps: specify the decoding fps of the video
        clip_length: the number of frames
        jitter: True stands for random sampling, False means center sampling
    return:
        frames: torch tensor with shape: [T, H, W, C]
    '''
    ### get vr ###
    if chunk_len == -1:
        if not vid.endswith('.mp4') and not vid.endswith('.mkv') and not vid.endswith('webm'):
            vid = vid + '.mp4'
        vr = get_vr(osp.join(root, vid))

        ### add a sanity check ###
        second = min(second, len(vr) / vr.get_avg_fps())

        second_offset = second
        if end_second is not None:
            end_second = min(end_second, len(vr) / vr.get_avg_fps())
            ### add a sanity check ###
            end_second = max(second + 1, end_second)
        else:
            end_second = len(vr) / vr.get_avg_fps()
    else:
        ### changed here to load chunked data ###
        chunk_id = int(second) // chunk_len
        chunk_start = chunk_id * chunk_len
        second_offset = second - chunk_start
        try:
            vr = get_vr(osp.join(root, vid, '{}.mp4'.format(chunk_id)))
        except:
            vr = get_vr(osp.join(root, vid, '0.mp4'))
    
    fps = vr.get_avg_fps() if fps == -1 else fps

    ### calculate frame_ids ###
    frame_offset = int(np.round(second_offset * fps))
    total_duration = max(int((end_second - second) * fps), clip_length)
    # print(f'Frame offset: {frame_offset}, total_duration: {total_duration}, fps: {fps}, start_second: {second}, end_second: {end_second}')

    if chunk_len == -1:
        if end_second <= second:
            # raise ValueError("end_second should be greater than second")
            ## changed here to not stop running
            print("end_second should be greater than second for video:{} from {}-{}".format(vid, second, end_second))
        
        #print(second, end_second, frame_offset, frame_offset + total_duration, len(vr))
        #set_trace()
        frame_ids = get_frame_ids(frame_offset, min(frame_offset + total_duration, len(vr)), num_segments=clip_length, jitter=jitter)
    else:
        # frame_ids = get_frame_ids(frame_offset, frame_offset + total_duration, num_segments=clip_length, jitter=jitter)
        frame_ids = get_frame_ids(frame_offset, min(frame_offset + total_duration, len(vr)), num_segments=clip_length, jitter=jitter)


    ### add a sanity check for the frame indices in HowTo100M ###
    if 'howto100' in root and max(frame_ids) >= len(vr):
        print(f'Selecting video {vid} with frames larger than the end')
        frame_ids = [min(x, len(vr) - 1) for x in frame_ids]

    ### load frames ###
    if max(frame_ids) < len(vr):
        try:
            frames = vr.get_batch(frame_ids).asnumpy()
        except decord.DECORDError as error:
            print(error)
            frames = vr.get_batch([0] * len(frame_ids)).asnumpy()
    else:
        # find the remaining frames in the next chunk
        try:
            frame_ids_part1 = list(filter(lambda frame_id: frame_id < len(vr), frame_ids))
            frames_part1 = vr.get_batch(frame_ids_part1).asnumpy()
            # vr2 = decord.VideoReader(osp.join(root, '{}.mp4'.format(vid), '{}.mp4'.format(chunk_start + chunk_len)))
            ### changed here ###
            if os.path.exists(osp.join(root, vid, '{}.mp4'.format(chunk_id + 1))):
                vr2 = get_vr(osp.join(root, vid, '{}.mp4'.format(chunk_id + 1)))
            else:
                vr2 = vr
            
            frame_ids_part2 = list(filter(lambda frame_id: frame_id >= len(vr), frame_ids))
            frame_ids_part2 = [min(frame_id % len(vr), len(vr2) - 1) for frame_id in frame_ids_part2]
            frames_part2 = vr2.get_batch(frame_ids_part2).asnumpy()
            frames = np.concatenate([frames_part1, frames_part2], axis=0)
        # the next chunk does not exist; the current chunk is the last one
        except (RuntimeError, decord.DECORDError) as error:
            print(error)
            frame_ids = get_frame_ids(min(frame_offset, len(vr) - 1), len(vr), num_segments=clip_length, jitter=jitter)
            frames = vr.get_batch(frame_ids).asnumpy()

    frames = [torch.tensor(frame, dtype=torch.float32) for frame in frames]
    return torch.stack(frames, dim=0)

def get_frame_ids(start_frame, end_frame, num_segments=32, jitter=True):
    '''
    args:
        start_frame: the beginning frame indice
        end_frame: the end frame indice
        num_segment: number of frames to be sampled
        jitter: True stands for random sampling, False means center sampling
    return:
        seq: a list for the sampled frame indices 
    '''
    assert start_frame <= end_frame

    seg_size = float(end_frame - start_frame - 1) / num_segments
    seq = []
    for i in range(num_segments):
        start = int(np.round(seg_size * i) + start_frame)
        end = int(np.round(seg_size * (i + 1)) + start_frame)
        # end = min(end, end_frame)
        
        ### added here to avoid out-of-boundary of frame_id, as np.random.randint ###
        start = min(start, end_frame-1)
        end = min(end, end_frame)

        if jitter:
            frame_id = np.random.randint(low=start, high=(end + 1))
        else:
            frame_id = (start + end) // 2
        # print(start_frame, end_frame, start, end, end_frame, frame_id)
        seq.append(frame_id)
    return seq

def video_loader_by_frames(root, vid, frame_ids):
    '''
    args:
        root: root directory of the video
        vid: the unique vid of the video, e.g. hello.mp4 
        frame_ids: the sampled frame indices 
    return:
        frames: torch tensor with shape: [T, H, W, C]
    '''
    vr = get_vr(osp.join(root, vid))
    try:
        frames = vr.get_batch(frame_ids).asnumpy()
        frames = [torch.tensor(frame, dtype=torch.float32) for frame in frames]
    except (IndexError, decord.DECORDError) as error:
        print(error)
        print("Erroneous video: ", vid)
        frames = [torch.zeros((240, 320, 3)) for _ in range(len(frame_ids))]
    return torch.stack(frames, dim=0) 

def video_loader_by_timestamp(root, vid, start_timestamp=0, end_timestamp=0, clip_length=4, is_training=False):
    '''
    args:
        root: root directory of the video
        vid: the unique vid of the video, e.g. hello.mp4 
        start_timestamp: the start second of the clip/video
        end_timestamp: the end second of the clip/video
        clip_length: the number of frames to be sampled
        is_training: whether it is training, jitter=True/False for train/test
    return:
        frames: torch tensor with shape: [T, H, W, C]
    '''
    ## change calculate start/end frame here
    vr = get_vr(osp.join(root, vid))
    fps = vr.get_avg_fps()

    ### this is for 320p ###
    start_frame = int(np.round(fps * start_timestamp)) if start_timestamp else 0
    end_frame = int(np.ceil(fps * end_timestamp)) if end_timestamp else len(vr) - 1
    frame_ids = get_frame_ids(start_frame, end_frame, num_segments=clip_length, jitter=is_training)
    
    try:
        frames = vr.get_batch(frame_ids).asnumpy()
        frames = [torch.tensor(frame, dtype=torch.float32) for frame in frames]
    except (IndexError, decord.DECORDError) as error:
        print(error)
        print("Erroneous video: ", vid)
        frames = [torch.zeros((240, 320, 3)) for _ in range(len(frame_ids))]
    return torch.stack(frames, dim=0)

def video_loader_by_timestamp_centerframe(root, vid, start_timestamp=0, end_timestamp=0, clip_length=4, is_training=False):
    '''
    args:
        root: root directory of the video
        vid: the unique vid of the video, e.g. hello.mp4 
        start_timestamp: the start second of the clip/video
        end_timestamp: the end second of the clip/video
        clip_length: the number of frames to be sampled
        is_training: whether it is training, jitter=True/False for train/test
    return:
        frames: torch tensor with shape: [T, H, W, C]
    '''
    ## change calculate start/end frame here
    vr = get_vr(osp.join(root, vid))
    fps = vr.get_avg_fps()

    ### this is for 320p ###
    start_frame = int(np.round(fps * start_timestamp)) if start_timestamp else 0
    end_frame = int(np.ceil(fps * end_timestamp)) if end_timestamp else len(vr) - 1
    frame_ids = get_frame_ids(start_frame, end_frame, num_segments=clip_length, jitter=is_training)
    frame_ids = frame_ids[len(frame_ids)//2 - 1: len(frame_ids)//2]
    
    try:
        frames = vr.get_batch(frame_ids).asnumpy()
        frames = [torch.tensor(frame, dtype=torch.float32) for frame in frames]
    except (IndexError, decord.DECORDError) as error:
        print(error)
        print("Erroneous video: ", vid)
        frames = [torch.zeros((240, 320, 3)) for _ in range(len(frame_ids))]
    return torch.stack(frames, dim=0)


def video_loader_by_array(root, vid, frame_ids):
    '''
    Directly load the array into the memory without decoding the video
    args:
        root: root directory of the video
        vid: the unique vid of the video, e.g. hello.mp4 
        frame_ids: the sampled frame indices 
    return:
        frames: torch tensor with shape: [T, H, W, C]
    '''

    video_path = os.path.join(root, vid)
    video_bytes = io.BytesIO(client.get(video_path))
    tensor = torch.load(video_bytes)
    pass

class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width  # 14x14
        self.total_patches = self.frames * self.num_patches_per_frame
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Tube Masking: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks)
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames, 1))
        return mask  # [196*8]


def generate_label_map(dataset, metapath):
    if dataset == 'ek100_cls':
        print("Preprocess ek100 action label space")
        vn_list = []
        mapping_vn2narration = {}
        for f in [
            # '/mnt/petrelfs/xujilan/data/epic_kitchen/epic-kitchens-100-annotations/EPIC_100_train.csv',
            # '/mnt/petrelfs/xujilan/data/epic_kitchen/epic-kitchens-100-annotations/EPIC_100_validation.csv',
            f'{metapath}epic-kitchens-100-annotations/EPIC_100_train.csv',
            f'{metapath}epic-kitchens-100-annotations/EPIC_100_validation.csv',
        ]:
            csv_reader = csv.reader(open(f))
            _ = next(csv_reader)  # skip the header
            for row in csv_reader:
                vn = '{}:{}'.format(int(row[10]), int(row[12]))
                narration = row[8]
                if vn not in vn_list:
                    vn_list.append(vn)
                if vn not in mapping_vn2narration:
                    mapping_vn2narration[vn] = [narration]
                else:
                    mapping_vn2narration[vn].append(narration)
                # mapping_vn2narration[vn] = [narration]
        vn_list = sorted(vn_list)
        print('# of action= {}'.format(len(vn_list)))
        mapping_vn2act = {vn: i for i, vn in enumerate(vn_list)}
        labels = [list(set(mapping_vn2narration[vn_list[i]])) for i in range(len(mapping_vn2act))]
        print(labels[:5])
    elif dataset == 'charades_ego':
        print("=> preprocessing charades_ego action label space")
        vn_list = []
        labels = []
        # with open('datasets/CharadesEgo/CharadesEgo/Charades_v1_classes.txt') as f:
        with open(f'{metapath}Charades_v1_classes.txt') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                vn = row[0][:4]
                vn_list.append(vn)
                narration = row[0][5:]
                labels.append(narration)
        mapping_vn2act = {vn: i for i, vn in enumerate(vn_list)}
        print(labels[:5])
    elif dataset == 'egtea':
        print("=> preprocessing egtea action label space")
        labels = []
        # with open('datasets/EGTEA/action_idx.txt') as f:
        with open(f'{metapath}action_idx.txt') as f:
            for row in f:
                row = row.strip()
                narration = ' '.join(row.split(' ')[:-1])
                labels.append(narration.replace('_', ' ').lower())
                # labels.append(narration)
        mapping_vn2act = {label: i for i, label in enumerate(labels)}
        print(len(labels), labels[:5])
    else:
        raise NotImplementedError
    return labels, mapping_vn2act

################## For positive pair construction ##################
def get_vector(noun_idx, verb_idx):
    noun_dim = 582
    verb_dim = 118
    noun_vec = torch.zeros(noun_dim)
    verb_vec = torch.zeros(verb_dim)
    noun_vec[noun_idx] = 1
    verb_vec[verb_idx] = 1
    all_vec = torch.cat([noun_vec, verb_vec])
    return all_vec    

def build_feature_matrix(rule = 'nounverb', egoexo_flag = 0):
    if rule == 'nounverb':
        if egoexo_flag == 0:
            metadir = '/mnt/petrelfs/xujilan/data/ego4d/generated/ego4d_train_nounverb_v1.json'
            savedir = '/mnt/petrelfs/xujilan/data/ego4d/generated/ego4d_train_nounverb_v1_nvfeat.pth'
        else:
            metadir = '/mnt/petrelfs/xujilan/data/howto100/generated/htm_aa_v2_food_clips_nounverb_v1.json'
            savedir = '/mnt/petrelfs/xujilan/data/howto100/generated/htm_aa_v2_food_clips_nounverb_v1_nvfeat.pth'

        metadata = json.load(open(metadir))
        all_feat = []
        for i in range(len(metadata)):
            noun_idx, verb_idx = metadata[str(i)]['noun'], metadata[str(i)]['verb']
            curr_vec = get_vector(noun_idx, verb_idx)
            all_feat.append(curr_vec)
            if (i % 10000 == 0):
                print(f'Done {i}/{len(metadata)}')
            #break
        all_feat = torch.stack(all_feat)
        print(all_feat.shape)
        torch.save(all_feat, savedir)
    else:
        pass


def construct_positive_pairs(rule = 'nounverb', egoexo_flag = 0):
    import faiss
    K = 16
    if rule == 'nounverb':
        dimension = 582 + 118 
        mat1 = torch.load('/mnt/petrelfs/xujilan/data/ego4d/generated/ego4d_train_nounverb_v1_nvfeat.pth')
        mat2 = torch.load('/mnt/petrelfs/xujilan/data/howto100/generated/htm_aa_v2_food_clips_nounverb_v1_nvfeat.pth')
    else:
        dim = 1536
        mat1 = None
        mat2 = None

    if egoexo_flag == 1:
        mat1, mat2 = mat2, mat1
        
    index = faiss.IndexFlatL2(dimension)
    index.add(mat1)
    distances, indices = index.search(mat2, K)
    st()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('--flag', default=0, type=int)
    args = parser.parse_args()
    # build_feature_matrix(egoexo_flag=args.flag)
    construct_positive_pairs(egoexo_flag=args.flag)
