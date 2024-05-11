
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
from ipdb import set_trace
import torch.nn as nn

class Permute(nn.Module):
    """
    Permutation as an op
    """

    def __init__(self, ordering):
        super().__init__()
        self.ordering = ordering

    def forward(self, frames):
        """
        Args:
            frames in some ordering, by default (C, T, H, W)
        Returns:
            frames in the ordering that was specified
        """
        return frames.permute(self.ordering)

def get_vr(client, video_path):
    if client is None:
        vreader = decord.VideoReader(video_path, ctx=cpu(0))
        
    else:        
        video_bytes = client.get(video_path, enable_stream=True)
        assert video_bytes is not None, "Get video failed from {}".format(video_path)
        video_path = video_bytes
        if isinstance(video_path, bytes):
            video_path = io.BytesIO(video_bytes)
        vreader = decord.VideoReader(video_path, ctx=cpu(0))
        
    return vreader   
    
def video_loader(client, root, vid, second=None, end_second=None, chunk_len=300, fps=30, clip_length=32, jitter=False):
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
        vr = get_vr(client, osp.join(root, vid))

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
            vr = get_vr(client, osp.join(root, vid, '{}.mp4'.format(chunk_id)))
        except:
            vr = get_vr(client, osp.join(root, vid, '0.mp4'))
    
    fps = vr.get_avg_fps() if fps == -1 else fps

    ### calculate frame_ids ###
    frame_offset = int(np.round(second_offset * fps))
    total_duration = max(int((end_second - second) * fps), clip_length)

    if chunk_len == -1:
        if end_second <= second:
            print("end_second should be greater than second for video:{} from {}-{}".format(vid, second, end_second))
            
        frame_ids = get_frame_ids(frame_offset, min(frame_offset + total_duration, len(vr)), num_segments=clip_length, jitter=jitter)
    else:
        frame_ids = get_frame_ids(frame_offset, frame_offset + total_duration, num_segments=clip_length, jitter=jitter)

    ### add a sanity check for the frame indices ###
    if max(frame_ids) >= len(vr):
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

            if os.path.exists(osp.join(root, vid, '{}.mp4'.format(chunk_id + 1))):
                vr2 = get_vr(client, osp.join(root, vid, '{}.mp4'.format(chunk_id + 1)))
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
        
        start = min(start, end_frame-1)
        end = min(end, end_frame)

        if jitter:
            frame_id = np.random.randint(low=start, high=(end + 1))
        else:
            frame_id = (start + end) // 2
        seq.append(frame_id)
    return seq
