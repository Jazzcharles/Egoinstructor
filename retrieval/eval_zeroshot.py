# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import numpy as np
import os.path as osp
import time
from collections import OrderedDict

import pandas as pd
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

import torchvision.transforms._transforms_video as transforms_video
from sklearn.metrics import confusion_matrix
from function import distributed as dist_utils
from function.meter import accuracy, get_mean_accuracy, egomcq_accuracy_metrics, calculate_k_counts, calculate_IDCG, calculate_mAP, calculate_nDCG, charades_map, get_mAP, get_nDCG, compute_metrics

import clip
from function.func_utils import gather_obj, gather, generate_tokenizer
from models.builder import build_model

def get_args_parser():
    parser = argparse.ArgumentParser(description='EgoInstructor 0-shot evaluations', add_help=False)
    parser.add_argument('--dataset', default='ek100_mir', type=str,
                        choices=['ek100_cls', 'ek100_mir', 'charades_ego', 'egtea', 'ego4d_mcq'])
    parser.add_argument('--root',
                        default='datasets/EK100/video_ht256px/',
                        type=str, help='path to dataset root')
    parser.add_argument('--metadata-val',
                        default='datasets/EK100/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_test.csv',
                        type=str, help='path to metadata file (val set)')
    parser.add_argument('--relevancy-path',
                        default='datasets/EK100/epic-kitchens-100-annotations/retrieval_annotations/relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl',
                        type=str, help='path to relevancy matrix (val set)')
    parser.add_argument('--output-dir', default='./', type=str, help='output dir')
    parser.add_argument('--num-crops', default=1, type=int, help='number of crops in transforms')
    parser.add_argument('--num-clips', default=1, type=int, help='number of clips (for untrimmed videos, eg. Charades)')
    parser.add_argument('--clip-length', default=4, type=int, help='clip length')
    parser.add_argument('--clip-stride', default=16, type=int, help='clip stride')
    parser.add_argument('--sparse-sample', action='store_true', help='switch to sparse sampling')
    parser.add_argument('--batch-size', default=16, type=int, help='batch_size')
    parser.add_argument('--cls-use-template', action='store_true', help='use prompt in 0-shot classification')
    parser.add_argument('--print-freq', default=100, type=int)
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
    parser.add_argument('--use-half', action='store_true')
    return parser


def get_video2video_similarity_matrix(val_loader, model, print_freq=100, args=None, cfg=None):
    model.eval()
    if cfg.train.use_half:
        model = model.half()
    
    all_ego_video_embed = []
    all_exo_video_embed = []
    with torch.no_grad():
        print('=> encoding ego visual and exo visual')
        for i, inputs in enumerate(val_loader):
            if i % print_freq == 0:
                print('finish batch {}/{}'.format(i, len(val_loader)))
            ego_frames = inputs[0].cuda(non_blocking=True)
            if cfg.train.use_half:
                frames = frames.half()
            exo_frames = inputs[1].cuda(non_blocking=True)

            # encode images
            ego_image_features = dist_utils.get_model(model).encode_image(ego_frames)
            ego_image_features = ego_image_features / ego_image_features.norm(dim=-1, keepdim=True)
            all_ego_video_embed.append(ego_image_features.cpu().numpy())

            exo_image_features = dist_utils.get_model(model).encode_image(exo_frames)
            exo_image_features = exo_image_features / exo_image_features.norm(dim=-1, keepdim=True)
            all_exo_video_embed.append(exo_image_features.cpu().numpy())

        all_ego_video_embed = np.vstack(all_ego_video_embed)
        all_exo_video_embed = np.vstack(all_exo_video_embed)
        
        similarity_matrix = np.matmul(all_ego_video_embed, all_exo_video_embed.T)
        
    return similarity_matrix


def get_similarity_matrix(val_loader, model, print_freq=100, args=None, cfg=None):
    model.eval()
    if cfg.train.use_half:
        model = model.half()
    all_text_embed = []
    all_video_embed = []
    with torch.no_grad():
        print('=> encoding visual and textual')
        for i, inputs in enumerate(val_loader):
            if i % print_freq == 0:
                print('finish batch {}/{}'.format(i, len(val_loader)))


            frames = inputs['video'].cuda(non_blocking=True)
            if cfg.train.use_half:
                frames = frames.half()
            texts = inputs['text'].cuda(non_blocking=True)

            if 'mask' in inputs:
                masks = inputs['mask'].cuda(non_blocking=True)
            else:
                masks = None

            # encode images
            image_features = dist_utils.get_model(model).encode_image(frames)
            
            if texts.ndim == 3:
                is_multiple_narrations = True
                texts = texts.view(-1, texts.shape[-1])
            else:
                is_multiple_narrations = False
            text_features = dist_utils.get_model(model).encode_text(texts)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_video_embed.append(image_features.cpu().numpy())

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            all_text_embed.append(text_features.cpu().numpy())
        
        all_text_embed = np.vstack(all_text_embed)
        all_video_embed = np.vstack(all_video_embed)
        similarity_matrix = np.matmul(all_video_embed, all_text_embed.T)
        if is_multiple_narrations:
            similarity_matrix = similarity_matrix.reshape(all_video_embed.shape[0], all_video_embed.shape[0], -1)

    return similarity_matrix


def validate_mcq(val_loader, model, use_half=False, cfg=None, args=None):
    model.eval()
    if use_half:
        model.half()
    with torch.no_grad():
        print('=> start forwarding')
        all_preds = []
        all_gts = []
        all_types = []
        end_time = time.time()
        for i, inputs in enumerate(val_loader):
            if i % 10 == 0:
                print('finish batch {}/{} in {} sec'.format(i, len(val_loader), time.time() - end_time))
                end_time = time.time()
            texts_query = inputs[0].cuda(non_blocking=True)
            frames_options = inputs[1].cuda(non_blocking=True)
            if use_half:
                frames_options = frames_options.half()
            answer = inputs[3]
            q_type = inputs[4]
            if len(inputs) == 7:
                masks_query = inputs[5].cuda(non_blocking=True)
            else:
                masks_query = None

            batch_size = frames_options.shape[0]

            frames_options = frames_options.view(-1, *frames_options.shape[2:])

            image_features = dist_utils.get_model(model).encode_image(frames_options)
            image_features = image_features.view(batch_size, -1, *image_features.shape[1:])

            if masks_query is not None:
                query_features = dist_utils.get_model(model).encode_text(texts_query, attention_mask=masks_query)
            else:
                query_features = dist_utils.get_model(model).encode_text(texts_query)

            all_gts.append(answer)
            all_types.append(q_type)
            for j in range(batch_size):
                similarity_matrix = torch.matmul(query_features[j], image_features[j].T)
                similarity_matrix = similarity_matrix.cpu().detach()
                all_preds.append(similarity_matrix)
                
                
        all_preds = torch.stack(all_preds)
        all_gts = torch.cat(all_gts)
        all_types = torch.cat(all_types)

        metrics = egomcq_accuracy_metrics(all_preds, all_gts, all_types)
        return metrics


def validate_v2t_mcq(val_loader, model, use_half=False, cfg=None, args=None):
    model.eval()
    if use_half:
        model.half()
    with torch.no_grad():
        print('=> start forwarding')
        all_preds = []
        all_gts = []
        all_types = []
        end_time = time.time()
        for i, inputs in enumerate(val_loader):
            if i % 10 == 0:
                print('finish batch {}/{} in {} sec'.format(i, len(val_loader), time.time() - end_time))
                end_time = time.time()
            
            ### data loading ###
            frame_query = inputs[0].cuda(non_blocking=True)
            narration_options = inputs[1].cuda(non_blocking=True)

            if use_half:
                frame_query = frame_query.half()
            answer = inputs[2]
            q_type = inputs[3]
            batch_size = frame_query.shape[0]

            ### encode video ###
            image_features = dist_utils.get_model(model).encode_image(frame_query)

            ### encode narration ###
            narration_options = narration_options.view(-1, *narration_options.shape[2:])
            narration_options_features = dist_utils.get_model(model).encode_text(narration_options)
            
            narration_options_features = narration_options_features.view(batch_size, -1, *narration_options_features.shape[1:])


            image_features = F.normalize(image_features, dim=-1)
            narration_options_features = F.normalize(narration_options_features, dim=-1)

            all_gts.append(answer)
            all_types.append(q_type)
            for j in range(batch_size):
                similarity_matrix = torch.matmul(image_features[j], narration_options_features[j].T)
                similarity_matrix = similarity_matrix.cpu().detach()
                all_preds.append(similarity_matrix)
                
        all_preds = torch.stack(all_preds)
        all_gts = torch.cat(all_gts)
        all_types = torch.cat(all_types)
        metrics = egomcq_accuracy_metrics(all_preds, all_gts, all_types)
        return metrics
    
   
def validate_v2v_mcq(val_loader, model, use_half=False, cfg=None, args=None):
    model.eval()
    if use_half:
        model.half()
    with torch.no_grad():
        print('=> start forwarding')
        all_preds = []
        all_gts = []
        all_types = []
        end_time = time.time()
        for i, inputs in enumerate(val_loader):
            if i % 10 == 0:
                print('finish batch {}/{} in {} sec'.format(i, len(val_loader), time.time() - end_time))
                end_time = time.time()
            frame_query = inputs[0].cuda(non_blocking=True)
            frames_options = inputs[1].cuda(non_blocking=True)
            if use_half:
                frames_options = frames_options.half()
            answer = inputs[2]
            q_type = inputs[3]
            
            batch_size = frames_options.shape[0]
            frames_options = frames_options.view(-1, *frames_options.shape[2:])

            ### encode videos ###
            image_query_features = dist_utils.get_model(model).encode_image(frame_query)
            image_options_features = dist_utils.get_model(model).encode_image(frames_options)
            image_options_features = image_options_features.view(batch_size, -1, *image_options_features.shape[1:])

            image_query_features = F.normalize(image_query_features, dim=-1)
            image_options_features = F.normalize(image_options_features, dim=-1)

            all_gts.append(answer)
            all_types.append(q_type)

            for j in range(batch_size):
                similarity_matrix = torch.matmul(image_query_features[j], image_options_features[j].T)
                similarity_matrix = similarity_matrix.cpu().detach()
                all_preds.append(similarity_matrix)
            
                
        all_preds = torch.stack(all_preds)
        all_gts = torch.cat(all_gts)
        all_types = torch.cat(all_types)
        metrics = egomcq_accuracy_metrics(all_preds, all_gts, all_types)
        return metrics

def validate_retrieval_zeroshot(val_loader, model, retrieval_type='v2t', args=None, cfg=None):
    if retrieval_type == 'v2v':
        similarity_matrix = get_video2video_similarity_matrix(val_loader, model, args=args, cfg=cfg)
    else:
        similarity_matrix = get_similarity_matrix(val_loader, model, args=args, cfg=cfg)
    msrvtt_v2t = compute_metrics(similarity_matrix)
    msrvtt_t2v = compute_metrics(similarity_matrix.T)
    return msrvtt_v2t['R1'], msrvtt_v2t['R5'], msrvtt_v2t['R10'], msrvtt_v2t['MeanR'], msrvtt_v2t['MedianR'], msrvtt_t2v['R1'], msrvtt_t2v['R5'], msrvtt_t2v['R10'], msrvtt_t2v['MeanR'], msrvtt_t2v['MedianR']
    
def validate_ek100_mir_zeroshot(ek100_loader, model, args, cfg):
    similarity_matrix = get_similarity_matrix(ek100_loader, model, args=args, cfg=cfg)
    similarity_matrix = (similarity_matrix + 1) / 2
    video_id = pd.read_csv("/mnt/petrelfs/xujilan/data/epic_kitchen/EPIC_100_retrieval_test.csv").values[:, 0]
    text_id = pd.read_csv("/mnt/petrelfs/xujilan/data/epic_kitchen/EPIC_100_retrieval_test_sentence.csv").values[:, 0]
    indexes = [video_id.tolist().index(elem) for elem in text_id]
    similarity_matrix = similarity_matrix[:, indexes]
    rel_matrix = pd.read_pickle(
        '/mnt/petrelfs/xujilan/data/epic_kitchen/relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl'
    )
    vis_map, txt_map, avg_map = get_mAP(similarity_matrix, rel_matrix)
    vis_ndcg, txt_ndcg, avg_ndcg = get_nDCG(similarity_matrix, rel_matrix)
    return vis_map, txt_map, avg_map, vis_ndcg, txt_ndcg, avg_ndcg

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser('EgoInstructor 0-shot evaluations', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
