
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

def get_video_reader(client, videoname, num_threads, fast_rrc, rrc_params, fast_rcc, rcc_params):
    video_reader = None
    video_bytes = client.get(videoname)
    assert video_bytes is not None, "Get video failed from {}".format(videoname)
    videoname = video_bytes
    
    if isinstance(videoname, bytes):
        videoname = io.BytesIO(video_bytes)
        
    video_reader = decord.VideoReader(videoname, num_threads=num_threads)
    return video_reader

def video_loader(client, root, vid, second, end_second, ext='mp4',
                 chunk_len=300, fps=-1, clip_length=32,
                 threads=1,
                 fast_rrc=False, rrc_params=(224, (0.5, 1.0)),
                 fast_rcc=False, rcc_params=(224, ),
                 jitter=False):
    # assert fps > 0, 'fps should be greater than 0'
    
    if chunk_len == -1:
        vr = get_video_reader(
            client,
            osp.join(root, '{}.{}'.format(vid, ext)),
            num_threads=threads,
            fast_rrc=fast_rrc, rrc_params=rrc_params,
            fast_rcc=fast_rcc, rcc_params=rcc_params,
        )
        fps = vr.get_avg_fps() if fps == -1 else fps
        
        end_second = min(end_second, len(vr) / fps)

        # calculate frame_ids
        frame_offset = int(np.round(second * fps))
        total_duration = max(int((end_second - second) * fps), clip_length)
        frame_ids = get_frame_ids(frame_offset, min(frame_offset + total_duration, len(vr)), num_segments=clip_length, jitter=jitter)
        
        # print(second, end_second, frame_ids)
        
        # load frames
        assert max(frame_ids) < len(vr)
        try:
            frames = vr.get_batch(frame_ids).asnumpy()
        except decord.DECORDError as error:
            print(error)
            frames = vr.get_batch([0] * len(frame_ids)).asnumpy()
    
        return torch.from_numpy(frames.astype(np.float32))

    else:
        assert fps > 0, 'fps should be greater than 0'
        
        ## sanity check, for those who have start >= end ##
        end_second = max(end_second, second + 1)

        chunk_start = int(second) // chunk_len * chunk_len
        chunk_end = int(end_second) // chunk_len * chunk_len
        
        # print(f'Vid={vid}, begin_sec={second}, end_sec={end_second}, \t, st_frame={int(np.round(second * fps))}, ed_frame={int(np.round(end_second * fps))}')
        # calculate frame_ids
        frame_ids = get_frame_ids(
            int(np.round(second * fps)),
            int(np.round(end_second * fps)),
            num_segments=clip_length, jitter=jitter
        )
        # print(f'Frames: {frame_ids}')
        
        all_frames = []
        # allocate absolute frame-ids into the relative ones
        for chunk in range(chunk_start, chunk_end + chunk_len, chunk_len):
            # print(f'Chunk: {chunk}, \t, Rel_frame_ids={rel_frame_ids}')
            vr = get_video_reader(
                client,
                # osp.join(root, '{}.{}'.format(vid, ext), '{}.{}'.format(chunk, ext)),
                # osp.join(root, f'{vid}', f'{chunk // chunk_len}.{ext}'),
                osp.join(root, vid, '{}.{}'.format(str(chunk // chunk_len).zfill(4), ext)),
                num_threads=threads,
                fast_rrc=fast_rrc, rrc_params=rrc_params,
                fast_rcc=fast_rcc, rcc_params=rcc_params,
            )

            rel_frame_ids = list(filter(lambda x: int(chunk * fps) <= x < int((chunk + chunk_len) * fps), frame_ids))
            # rel_frame_ids = [int(frame_id - chunk * fps) for frame_id in rel_frame_ids]
            rel_frame_ids = [min(len(vr) - 1, int(frame_id - chunk * fps)) for frame_id in rel_frame_ids]

            try:
                frames = vr.get_batch(rel_frame_ids).asnumpy()
            except decord.DECORDError as error:
                # print(error)
                frames = vr.get_batch([0] * len(rel_frame_ids)).asnumpy()
            except IndexError:
                print(root, vid, str(chunk // chunk_len).zfill(4), second, end_second)
                print(len(vr), rel_frame_ids)
            
            all_frames.append(frames)
            if sum(map(lambda x: x.shape[0], all_frames)) == clip_length:
                break
            
        res = torch.from_numpy(np.concatenate(all_frames, axis=0).astype(np.float32))
        assert res.shape[0] == clip_length, "{}, {}, {}, {}, {}, {}, {}".format(root, vid, second, end_second, res.shape[0], rel_frame_ids, frame_ids)
        return res
    
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
