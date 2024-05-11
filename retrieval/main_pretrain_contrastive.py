# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from collections import OrderedDict
import json
import math
import os
import pandas as pd
import sys
import time
import pickle

import torch
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.nn.parallel
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
import wandb
import numpy as np
import cv2
import logging
from copy import deepcopy
from omegaconf import OmegaConf, read_write
from mmcv.utils import collect_env, get_git_hash
from einops import rearrange
from sentence_transformers import SentenceTransformer

from eval_zeroshot import get_similarity_matrix, validate_ek100_mir_zeroshot, validate_retrieval_zeroshot, validate_mcq, validate_v2v_mcq

from data.video_transforms import Permute
from models.builder import build_model
from models import model_utils
from models import retriever

from function.meter import AverageMeter, ProgressMeter, compute_metrics, get_mAP, get_nDCG
from function import distributed as dist_utils

from function.scheduler import cosine_scheduler
from function.func_utils import build_train_loader, build_val_loader, build_optimizer, resume_checkpoint, build_scheduler, data_visualization, generate_tokenizer, random_seed
from function.config import get_config
from function.logger import get_logger

from ipdb import set_trace

def get_args_parser():
    parser = argparse.ArgumentParser(description='EgoInstructor training and evaluation', add_help=False)
    # Data
    parser.add_argument('--config', default='configs/default.yml', type=str)
    parser.add_argument('--dataset', default='ego4d', type=str, choices=['ego4d','howto100', 'ego4d_howto100'])
    parser.add_argument('--root', default='/mnt/petrelfs/share_data/chenguo/all_videos_fps30_short320_chunked/',
                        type=str, help='path to dataset root')
    parser.add_argument('--metadata', default='datasets/Ego4D/ego4d_train.pkl',
                        type=str, help='path to metadata file')
    parser.add_argument('--metadata-aux', default=None, nargs='+',
                        type=str, help='path to metadata file (auxiliary data with pseudo narrations)')
    parser.add_argument('--output', type=str, help='output dir')
    # Model
    parser.add_argument('--resume', default='', type=str, help='path to resume from')
    
    # Training
    parser.add_argument('--batch-size', default=32, type=int,
                        help='number of samples per-device/per-gpu')
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('--fix-lr', action='store_true', help='disable cosine lr decay if set True')
    parser.add_argument('--lr-start', default=1e-6, type=float,
                        help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-5, type=float,
                        help='minimum final lr')
    # System
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--wandb', action='store_true', help='Enable WandB logging')
   
    ### added ###
    parser.add_argument('--testonly', action='store_true', help='Enable zeroshot test only')
    return parser


def main(args):
    ##################################################### Prepare  ENV #####################################################
    cfg = get_config(args)
    os.makedirs(cfg.output, exist_ok=True)
    
    dist_utils.init_distributed_mode(args)
    logger = get_logger(cfg)   
    
    ### save config file ###
    if dist_utils.get_rank() == 0:
        path = os.path.join(cfg.output, 'config.json')
        OmegaConf.save(cfg, path)
        logger.info(f'Full config save to {path}')
    
    ### log config ###
    logger.info(OmegaConf.to_yaml(cfg))
    global best_acc1
    random_seed(cfg.train.seed, dist_utils.get_rank())
    ##################################################### Prepare Model #####################################################
    logger.info(f'Creating model:{cfg.model.name}')
    model = build_model(cfg.model)

    if cfg.model.freeze_temperature:
        logger.info('Freeze logit temperature')
        if hasattr(model, 'logit_scale'):
            model.logit_scale.requires_grad = False

    model.cuda(args.gpu)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], bucket_cap_mb=200,
            find_unused_parameters=cfg.train.find_unused_parameters
        )
    tokenizer = generate_tokenizer(cfg.model.name)    

    ##################################################### Prepare Optimizer &&  Criterion #####################################################
    criterion = model_utils.get_loss(cfg.model.name, args, cfg, tokenizer=tokenizer).cuda(args.gpu)
    optimizer = build_optimizer(cfg.train, model, criterion)
    scaler = amp.GradScaler(enabled=not cfg.train.disable_amp)
    lr_schedule = build_scheduler(cfg)
    
    ####################################### Auto Resume #######################################
    # optionally resume from a checkpoint (takes precedence over autoresume)
    loaded_resume = resume_checkpoint(cfg, model, optimizer, scaler, criterion)
    start_epoch, best_acc1 = loaded_resume['start_epoch'], loaded_resume['best_acc1']
    cudnn.benchmark = True

    ##################################################### Prepare dataset #####################################################
    logger.info("=> creating dataset")
    train_loader, train_sampler = build_train_loader(args, cfg, tokenizer)

    youcook_loader = build_val_loader(args, cfg, dataset_name='youcook', tokenizer=deepcopy(tokenizer))
    ek100_loader = build_val_loader(args, cfg, dataset_name='ek100_mir', tokenizer=deepcopy(tokenizer))
    egolearn_v2v_loader = build_val_loader(args, cfg, dataset_name='egolearn', tokenizer=deepcopy(tokenizer))
    # egomcq_loader =  build_val_loader(args, cfg, dataset_name='egomcq',tokenizer=deepcopy(tokenizer))

    ### long-term loader ###
    youcook_video_loader = build_val_loader(args, cfg, dataset_name='youcook_video', tokenizer=deepcopy(tokenizer))
    egosumm_loader = build_val_loader(args, cfg, dataset_name='egosumm', tokenizer=deepcopy(tokenizer))
    charades_egoexo_loader = build_val_loader(args, cfg, dataset_name='charades_egoexo', tokenizer=deepcopy(tokenizer))
    ##################################################### Prepare others #####################################################
    
    if dist_utils.is_main_process() and cfg.wandb:
        wandb_id = os.path.split(cfg.output)[-1]
        wandb.init(project='egoexo', id=wandb_id, config=args, resume='allow')

    if cfg.test.testonly:    
        ### ego video-text retrieval ###
        vis_map, txt_map, avg_map, vis_ndcg, txt_ndcg, avg_ndcg = validate_ek100_mir_zeroshot(ek100_loader, model, args=args, cfg=cfg)
        logger.info('EK100 MIR mAP: V->T: {:.3f} T->V: {:.3f} AVG: {:.3f}'.format(vis_map, txt_map, avg_map))
        logger.info('EK100 MIR nDCG: V->T: {:.3f} T->V: {:.3f} AVG: {:.3f}'.format(vis_ndcg, txt_ndcg, avg_ndcg))
        
        # # ### exo video-text retrieval ###
        R1_v2t, R5_v2t, R10_v2t, MeanR_v2t, MedianR_v2t, R1_t2v, R5_t2v, R10_t2v, MeanR_t2v, MedianR_t2v = validate_retrieval_zeroshot(youcook_loader, model, args=args, cfg=cfg)
        logger.info('YouCook Exo Retrieval_result V->T: R1={}, R5={}, R10={}, MeanR={}, MedianR={}'.format(R1_v2t, R5_v2t, R10_v2t, MeanR_v2t, MedianR_v2t))
        logger.info('YouCook Exo Retrieval_result T->V: R1={}, R5={}, R10={}, MeanR={}, MedianR={}'.format(R1_t2v, R5_t2v, R10_t2v, MeanR_t2v, MedianR_t2v))
        
        ### ego multiple-choice, testing takes long-time ###
        # metrics = validate_mcq(egomcq_loader, model, use_half=False, cfg=cfg, args=args)
        # print(metrics)
        # print('#' * 100)

        ### egolearn ego video-exo video retrieval ###
        metrics = validate_v2v_mcq(egolearn_v2v_loader, model, use_half=False, cfg=cfg, args=args)
        print(metrics)
        print('#' * 100)
        
        ### charades ego video-exo video retrieval ###
        R1_v2t, R5_v2t, R10_v2t, MeanR_v2t, MedianR_v2t, R1_t2v, R5_t2v, R10_t2v, MeanR_t2v, MedianR_t2v = validate_retrieval_zeroshot(charades_egoexo_loader, model, retrieval_type='v2v', args=args, cfg=cfg)
        logger.info('Charades Ego2Exo Retrieval_result R1={}, R5={}, R10={}, MeanR={}, MedianR={}'.format(R1_v2t, R5_v2t, R10_v2t, MeanR_v2t, MedianR_v2t))
        logger.info('Charades Exo2Ego Retrieval_result R1={}, R5={}, R10={}, MeanR={}, MedianR={}'.format(R1_t2v, R5_t2v, R10_t2v, MeanR_t2v, MedianR_t2v))
        
        ###### long-term evaluation ######
        ### exo video-text retrieval ###
        R1_v2t, R5_v2t, R10_v2t, MeanR_v2t, MedianR_v2t, R1_t2v, R5_t2v, R10_t2v, MeanR_t2v, MedianR_t2v = validate_retrieval_zeroshot(youcook_video_loader, model, args=args, cfg=cfg)
        logger.info('Long-Term YouCook Exo Retrieval_result V->T: R1={}, R5={}, R10={}, MeanR={}, MedianR={}'.format(R1_v2t, R5_v2t, R10_v2t, MeanR_v2t, MedianR_v2t))
        logger.info('Long-Term YouCook Exo Retrieval_result T->V: R1={}, R5={}, R10={}, MeanR={}, MedianR={}'.format(R1_t2v, R5_t2v, R10_t2v, MeanR_t2v, MedianR_t2v))
        
        ### ego-summary multiple-choice ###
        metrics = validate_mcq(egosumm_loader, model, use_half=False, cfg=cfg, args=args)
        print(metrics)
        print('#' * 100)

        return 
    
    best_metric = 0.
    print("=> beginning training")
    for epoch in range(start_epoch, cfg.train.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args, cfg, logger)
        
        ### logging training stats ###
        for k, v in train_stats.items():
            logger.info(f'Epoch {epoch}: Train_{k}: {round(v, 3)}')

        ### saving per epoch model ckpt before evaluation ###
        logger.info('=> saving per-epoch checkpoint')
        dist_utils.save_on_master({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'criterion': criterion.state_dict(),
            'optimizer': optimizer.state_dict() if dist_utils.get_rank() == 0 else {},
            'scaler': scaler.state_dict(),
            'best_acc1': best_metric,
            'cfg': cfg,
        }, False, cfg.output, is_epoch=True)

        ### exo retrieval evaluation ###
        logger.info('=> 0-shot on YouCook2 Retrieval')
        R1_v2t, R5_v2t, R10_v2t, MeanR_v2t, MedianR_v2t, R1_t2v, R5_t2v, R10_t2v, MeanR_t2v, MedianR_t2v = validate_retrieval_zeroshot(youcook_loader, model, args=args, cfg=cfg)
        logger.info('Exo_result V->T: R1={}, R5={}, R10={}, MeanR={}, MedianR={}'.format(R1_v2t, R5_v2t, R10_v2t, MeanR_v2t, MedianR_v2t))
        logger.info('Exo_result T->V: R1={}, R5={}, R10={}, MeanR={}, MedianR={}'.format(R1_t2v, R5_t2v, R10_t2v, MeanR_t2v, MedianR_t2v))
            
        ### ego retrieval evaluation ###
        logger.info('=> 0-shot on EK100 MIR')
        vis_map, txt_map, avg_map, vis_ndcg, txt_ndcg, avg_ndcg = validate_ek100_mir_zeroshot(ek100_loader, model, args=args, cfg=cfg)
        logger.info('mAP: V->T: {:.3f} T->V: {:.3f} AVG: {:.3f}'.format(vis_map, txt_map, avg_map))
        logger.info('nDCG: V->T: {:.3f} T->V: {:.3f} AVG: {:.3f}'.format(vis_ndcg, txt_ndcg, avg_ndcg))
        
        ### ego mcq evaluation,  it takes long time  ###
        # metrics = validate_mcq(egomcq_loader, model, use_half=False, cfg=cfg, args=args)
        # logger.info(f'EgoMCQ: Inter={metrics["Inter-video"]}, Intra={metrics["Intra-video"]}')

        ### egolearn evaluation ###
        logger.info('=> 0-shot on EgoLearn')
        v2v_metrics = validate_v2v_mcq(egolearn_v2v_loader, model, use_half=False, cfg=cfg, args=args)
        logger.info(f'V2V: Ego2Exo={v2v_metrics["Inter-video"]}, Exo2Ego={v2v_metrics["Intra-video"]}')
        
        ### charades-ego evaluation
        R1_v2t, R5_v2t, R10_v2t, MeanR_v2t, MedianR_v2t, R1_t2v, R5_t2v, R10_t2v, MeanR_t2v, MedianR_t2v = validate_retrieval_zeroshot(charades_egoexo_loader, model, retrieval_type='v2v', args=args, cfg=cfg)
        logger.info('Charades Ego2Exo Retrieval_result R1={}, R5={}, R10={}, MeanR={}, MedianR={}'.format(R1_v2t, R5_v2t, R10_v2t, MeanR_v2t, MedianR_v2t))
        logger.info('Charades Exo2Ego Retrieval_result R1={}, R5={}, R10={}, MeanR={}, MedianR={}'.format(R1_t2v, R5_t2v, R10_t2v, MeanR_t2v, MedianR_t2v))
        
        if avg_map > best_metric:
            is_best = True
            best_metric = avg_map
        else:
            is_best = False   

        ### long-term evaluation ###
        R1_v2t, R5_v2t, R10_v2t, MeanR_v2t, MedianR_v2t, R1_t2v, R5_t2v, R10_t2v, MeanR_t2v, MedianR_t2v = validate_retrieval_zeroshot(youcook_video_loader, model, args=args,cfg=cfg)
        logger.info('Long-Term YouCook Exo Retrieval_result V->T: R1={}, R5={}, R10={}, MeanR={}, MedianR={}'.format(R1_v2t, R5_v2t, R10_v2t, MeanR_v2t, MedianR_v2t))
        logger.info('Long-Term YouCook Exo Retrieval_result T->V: R1={}, R5={}, R10={}, MeanR={}, MedianR={}'.format(R1_t2v, R5_t2v, R10_t2v, MeanR_t2v, MedianR_t2v))
        
        metrics = validate_mcq(egosumm_loader, model, use_half=False, cfg=cfg, args=args)
        logger.info(f'Ego summ_mcq: {metrics["Inter-video"]}')
      
        ### save checkpoint ###
        is_epoch = ((epoch + 1) % cfg.train.save_freq) == 0

        if args.distributed and cfg.train.use_zero:
            logger.info("=> consolidating state_dict before saving (due to ZeRO)")
            optimizer.consolidate_state_dict()

        logger.info('=> saving the best checkpoint')
        dist_utils.save_on_master({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'criterion': criterion.state_dict(),
            'optimizer': optimizer.state_dict() if dist_utils.get_rank() == 0 else {},
            'scaler': scaler.state_dict(),
            'best_acc1': best_metric,
            'cfg': cfg,
        }, is_best, cfg.output, is_epoch=is_epoch)


def train_one_epoch(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args, cfg, logger):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = model_utils.get_metric_names(cfg)
    
    iters_per_epoch = len(train_loader) // cfg.train.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    for data_iter, inputs in enumerate(train_loader):
        optim_iter = data_iter // cfg.train.update_freq
                
        # measure data loading time
        data_time.update(time.time() - end)

        # update weight decay and learning rate according to their schedule
        it = iters_per_epoch * epoch + optim_iter  # global training iteration
        for k, param_group in enumerate(optimizer.param_groups):
            if lr_schedule is not None:
                param_group['lr'] = lr_schedule[it]
        
        batch_size = inputs['video'].size(0)
        # uids = inputs['uid'] + inputs['pair_uid']

        ### debug ###
        # print('Current keys: ', inputs.keys())
        # print('Video shape: ', inputs['video'].shape)
        # print('Text shape:', inputs['text'].shape)
        # print('Vid:', inputs['vid'])
        # print('Noun vector: ', inputs['noun_vec'].shape)
        # print('Verb vector: ', inputs['verb_vec'].shape)
        # print(len(inputs['raw_caption']), inputs['raw_caption'])
        # set_trace()

        ### keep part-of inputs ###
        video = inputs['video'].cuda(args.gpu)
        text  = inputs['text'].cuda(args.gpu)
        noun_vec = inputs['noun_vec'].cuda(args.gpu)
        verb_vec = inputs['verb_vec'].cuda(args.gpu)

        if len(video.size()) == 4:
            ### each sample has k cross-view pairs ###
            video = rearrange(video, 'b k t d -> (b k) t d')
            text = rearrange(text, 'b k d -> (b k) d')
            noun_vec = rearrange(noun_vec, 'b k d -> (b k) d')
            verb_vec = rearrange(verb_vec, 'b k d -> (b k) d')
            
        model_inputs = [video, text]
        # compute output
        with amp.autocast(enabled=not cfg.train.disable_amp):
            outputs = model(
                *model_inputs,
                use_checkpoint=cfg.train.use_checkpoint,
                norm_embed=cfg.model.norm_embed
            )     
            outputs['noun_vec'] = noun_vec
            outputs['verb_vec'] = verb_vec

            ### for debug only ###
            outputs['raw_caption'] = inputs['raw_caption'] 
               
            loss_dict = criterion(outputs)
            loss = loss_dict['loss']
            loss /= cfg.train.update_freq

        if not math.isfinite(loss.item()):
            logger.info("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        scaler.scale(loss).backward()

        if (data_iter + 1) % cfg.train.update_freq != 0:
            continue

        if cfg.train.clip_grad_value is not None:
            scaler.unscale_(optimizer)
            if cfg.train.clip_grad_type == 'norm':
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.train.clip_grad_value, norm_type=2.
                )
            elif cfg.train.clip_grad_type == 'value':
                torch.nn.utils.clip_grad_value_(model.parameters(), cfg.train.clip_grad_value)
            else:
                assert False, f"Unknown clip mode ({cfg.train.clip_grad_type})."
        # compute gradient and do SGD step
        scaler.step(optimizer)
        scaler.update()
        model.zero_grad(set_to_none=True)

        ### adjust logit scale ###
        if hasattr(dist_utils.get_model(model), 'logit_scale'):
            # clamp logit scale to [0, 100]
            dist_utils.get_model(model).logit_scale.data.clamp_(0, 4.6052)
            logit_scale = dist_utils.get_model(model).logit_scale.exp().item()
        else:
            logit_scale = torch.nan

        for k in loss_dict:
            metrics[k].update(loss_dict[k].item(), cfg.train.batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if optim_iter % cfg.train.print_freq == 0:
            if dist_utils.is_main_process():
                train_iter_log = {
                            'iter': data_iter,
                            **{k: round(v.item(), 3) for k, v in loss_dict.items()},
                           'scaler': round(scaler.get_scale(), 3), 
                           'logit': round(logit_scale, 3)}
                train_iter_log_str = ''
                for logk, logv in train_iter_log.items():
                    train_iter_log_str += f'{logk}:{logv}  '
                #set_trace()
                logger.info(train_iter_log_str)

            # progress.display(optim_iter)          
        #break
        
    progress.synchronize()
    return {**{k: v.avg for k, v in metrics.items()},
            'lr': optimizer.param_groups[0]['lr'],
            'logit_scale': logit_scale}
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('LaVid training and evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
