import torch
import torch.distributed as dist
import os, sys, subprocess
import json

from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.nn.parallel
import torchvision.transforms._transforms_video as transforms_video
import torchvision.transforms as transforms
import torch.nn.functional as F
import wandb
from typing import Tuple
import numpy as np
import cv2
import pickle
import random

from data.video_transforms import Permute, SpatialCrop, TemporalCrop
from .tokenizer import MyBertTokenizer, MyDistilBertTokenizer, SimpleTokenizer

from .scheduler import cosine_scheduler
from . import distributed as dist_utils
from .logger import get_logger
from data import *

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def gather(tensor, args):
    output = [torch.empty_like(tensor) for _ in range(args.world_size)]
    dist.all_gather(output, tensor)
    return torch.cat(output, 0)

def gather_obj(obj_list, args):
    output = [None for _ in range(args.world_size)]
    dist.all_gather_object(output, obj_list) 
    output = sum(output, []) ## convert the 2d list to 1d list
    return output

def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        #args.rank = int(os.environ['SLURM_PROCID'])
        #args.gpu = args.rank % torch.cuda.device_count()
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = os.environ['SLURM_NTASKS']
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list)
        )
        master_port = os.environ.get('MASTER_PORT', '29484')
        os.environ['MASTER_PORT'] = master_port
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.dist_url = 'env://'
        args.world_size = int(ntasks)
        args.rank = int(proc_id)
        args.gpu = int(proc_id % num_gpus)
        print(f'SLURM MODE: proc_id: {proc_id}, ntasks: {ntasks}, node_list: {node_list}, num_gpus:{num_gpus}, addr:{addr}, master port:{master_port}' )
        
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    
    args.distributed = True

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def convert_to_distributed_tensor(tensor: torch.Tensor) -> Tuple[torch.Tensor, str]:
    """
    For some backends, such as NCCL, communication only works if the
    tensor is on the GPU. This helper function converts to the correct
    device and returns the tensor + original device.
    """
    orig_device = "cpu" if not tensor.is_cuda else "gpu"
    if (
        torch.distributed.is_available()
        and torch.distributed.get_backend() == torch.distributed.Backend.NCCL
        and not tensor.is_cuda
    ):
        tensor = tensor.cuda()
    return (tensor, orig_device)


def convert_to_normal_tensor(tensor: torch.Tensor, orig_device: str) -> torch.Tensor:
    """
    For some backends, such as NCCL, communication only works if the
    tensor is on the GPU. This converts the tensor back to original device.
    """
    if tensor.is_cuda and orig_device == "cpu":
        tensor = tensor.cpu()
    return tensor


def is_distributed_training_run() -> bool:
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and (torch.distributed.get_world_size() > 1)
    )


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def gather_from_all(tensor: torch.Tensor) -> torch.Tensor:
    """
    Similar to classy_vision.generic.distributed_util.gather_from_all
    except that it does not cut the gradients
    """
    if tensor.ndim == 0:
        # 0 dim tensors cannot be gathered. so unsqueeze
        tensor = tensor.unsqueeze(0)

    if is_distributed_training_run():
        tensor, orig_device = convert_to_distributed_tensor(tensor)
        gathered_tensors = GatherLayer.apply(tensor)
        gathered_tensors = [
            convert_to_normal_tensor(_tensor, orig_device)
            for _tensor in gathered_tensors
        ]
    else:
        gathered_tensors = [tensor]
    gathered_tensor = torch.cat(gathered_tensors, 0)
    return gathered_tensor


def resume_checkpoint(cfg, model, optimizer, scaler, criterion):
    start_epoch = 0
    best_acc1 = 0.0
    latest = os.path.join(cfg.output, 'checkpoint.pt')
    use_latest = False
    if os.path.isfile(latest):
        # cfg.resume = ''
        use_latest = True
        
    logger = get_logger(cfg)
    #logger.root.handlers.clear()
    if use_latest:
        # auto-resume from latest checkpoint in output directory
        latest = os.path.join(cfg.output, 'checkpoint.pt')
        ### if checkpoint.pt does not exists, auto-resume the best-checkpoint ###
        latest = latest if os.path.isfile(latest) else latest.replace('checkpoint.pt', 'checkpoint_best.pt')

        if os.path.isfile(latest):
            logger.info("=> loading latest checkpoint '{}'".format(latest))
            latest_checkpoint = torch.load(latest, map_location='cpu')
            start_epoch = int(latest_checkpoint['epoch'])
            
            res = model.load_state_dict(latest_checkpoint['state_dict'])
            optimizer.load_state_dict(latest_checkpoint['optimizer'])
            
            logger.info('loading latest checkpoint:\n', res)
            scaler.load_state_dict(latest_checkpoint['scaler'])
            best_acc1 = latest_checkpoint['best_acc1']
            logger.info("=> loaded latest checkpoint '{}' (epoch {})"
                  .format(latest, latest_checkpoint['epoch']))
            # set_trace()
    elif cfg.resume:
        if os.path.isfile(cfg.resume):
            logger.info("=> loading resume checkpoint '{}'".format(cfg.resume))
            checkpoint = torch.load(cfg.resume, map_location='cpu')
            epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
            
            if 'state_dict' in checkpoint:
                ## VLCM models are saved by ourselves, i.e. with DDP weights always ###
                if 'positional_embedding' in checkpoint['state_dict']:
                    checkpoint['state_dict']['positional_embedding'] = interpolate_clip_positional_embeds(cfg.data.ctx_length, checkpoint['state_dict']['positional_embedding'])
                elif 'module.positional_embedding' in checkpoint['state_dict']:
                    checkpoint['state_dict']['module.positional_embedding'] = interpolate_clip_positional_embeds(cfg.data.ctx_length, checkpoint['state_dict']['module.positional_embedding'])

                result = model.load_state_dict(checkpoint['state_dict'], strict=False) 
            elif 'model' in checkpoint:
                result = model.load_state_dict(checkpoint['model'], strict=False)
            else:
                is_ddp_checkpoint = False
                for k, v in checkpoint.items():
                    if k.startswith('module.'):
                        is_ddp_checkpoint = True
                    break

                result = model.load_state_dict(checkpoint, strict=False) if is_ddp_checkpoint else model.module.load_state_dict(checkpoint, strict=False)
            
            logger.info(result)
            logger.info("=> loaded resume checkpoint '{}' (epoch {})".format(cfg.resume, epoch))
        else:
            print("=> no checkpoint found at '{}'".format(cfg.resume))
    else:
        print("=>No resumed checkpoint and no trained checkpoints")  
        
    return {
        'start_epoch': start_epoch,
        'best_acc1': best_acc1,
    }


def build_optimizer(cfg, model, criterion):
    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)

    for n, p in criterion.named_parameters():
        if not p.requires_grad:
            continue
        p_non_wd.append(p)

    optim_params = [{"params": p_wd, "weight_decay": cfg.optimizer.wd},
                    {"params": p_non_wd, "weight_decay": 0}]

    if cfg.use_zero:
        optimizer = ZeroRedundancyOptimizer(
            optim_params, optimizer_class=torch.optim.AdamW,
            lr=cfg.lr, betas=cfg.optimizer.betas, eps=cfg.optimizer.eps, weight_decay=cfg.optimizer.wd
        )
    else:
        optimizer = torch.optim.AdamW(optim_params, lr=cfg.lr, betas=cfg.optimizer.betas,
                                      eps=cfg.optimizer.eps, weight_decay=cfg.optimizer.wd)
    return optimizer


def generate_tokenizer(model):
    if model.endswith('DISTILBERT_BASE'):
        tokenizer = MyDistilBertTokenizer('distilbert-base-uncased')
    elif model.endswith('BERT_BASE'):
        tokenizer = MyBertTokenizer('bert-base-uncased')
    elif model.endswith('BERT_LARGE'):
        tokenizer = MyBertTokenizer('bert-large-uncased')
    else:
        print("Using SimpleTokenizer because of model '{}'. "
              "Please check if this is what you want".format(model))
        tokenizer = SimpleTokenizer()
    return tokenizer

def build_train_loader(args, cfg, tokenizer):
    train_dataset = EgoHowToFeatDataset(
        cfg=cfg.data, tokenizer=tokenizer, is_training=True, 
    )
       
    # print('loading data')
    # for i in range(1, 300000, 10000):
    #     t = train_dataset.__getitem__(i)
    #     # print(i, t['uid'], t['raw_caption'], t['second_ids'], sum(t['noun_vec']), sum(t['verb_vec']))
    #     print(i, t['uid'], t['raw_caption'])
    #     # print(i, t['noun_vec'].shape, t['verb_vec'].shape)
    #     # print(t['pair_raw_caption'])
    #     print()
    # st()
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.train.batch_size, shuffle=(train_sampler is None), # collate_fn = collate,
        num_workers=cfg.train.workers, pin_memory=True, sampler=train_sampler, drop_last=True
    )
    
    print('len(train_loader) = {}'.format(len(train_loader)))
    return train_loader, train_sampler

def build_val_loader(args, cfg, dataset_name='youcook', tokenizer=None):
    # TODO: uncomment when evaluation is done later
    if dataset_name == 'ek100_mir':
        val_dataset = EK100Dataset(
            cfg=cfg.test.ek100_mir,
            is_training=False,
            tokenizer=tokenizer,
        )
    elif dataset_name == 'youcook':
        val_dataset = YouCookDataset(
            cfg.test.youcook,
            is_training=False,
            tokenizer=tokenizer,
        )
    elif dataset_name == 'charades_egoexo':
        val_dataset = CharadesEgoExoDataset(
            cfg=cfg.test.charades_egoexo,
            is_training=False,
            tokenizer=tokenizer,
        )
    elif dataset_name == 'egomcq':
        val_dataset = EGOMCQDataset(
            cfg = cfg.test.egomcq,
            is_training=False,
            tokenizer=tokenizer,
        )
    elif dataset_name == 'youcook_video':
        val_dataset = YouCookDataset(
            cfg.test.youcook_video,
            is_training=False,
            tokenizer=tokenizer,
        )
    elif dataset_name == 'egosumm':
        val_dataset = EGOSUMMDataset(
            cfg = cfg.test.egosumm,
            is_training=False,
            tokenizer=tokenizer,
        )        
    elif dataset_name in ['egolearn']:
        val_dataset = EgoLearnDataset(
            cfg = cfg.test.egolearn,
            is_training=False,
            tokenizer=tokenizer,
        )
        
    val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.test.batch_size, shuffle=(val_sampler is None),
        num_workers=cfg.test.workers, pin_memory=True, sampler=val_sampler, drop_last=False,
    )
    
    #     for i in range(0, len(val_dataset), 1):
    #         temp = val_dataset.__getitem__(i)
    #         #print(i, temp['uid'], temp['start'], temp['end'], temp['raw_caption'])
    #         # print(i, temp[0], temp[1], temp[2])
    #         print(i)
    #         print(temp[-2], temp[-1])
    #         st()

    print('{} ==> len(val_dataset)={},len(val_dataloader)={}'.format(dataset_name, len(val_dataset), len(dataloader)))
    return dataloader
    

def build_scheduler(cfg):
    if cfg.train.fix_lr:
        lr_schedule = None
    else:
        lr_schedule = cosine_scheduler(
            cfg.train.lr, cfg.train.lr_end, cfg.train.epochs, len(train_loader) // cfg.train.update_freq,
            warmup_epochs=cfg.train.warmup_epochs, start_warmup_value=cfg.train.lr_start,
        )
    return lr_schedule

def write_log(args, train_stats, youcook_caption_log, ego4dcap_log, epoch):
    ### save evaluation results ###
    if dist_utils.is_main_process():
        if args.wandb:
            wandb.log(youcook_caption_log)
            wandb.log(ego4dcap_log)
        with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
            f.write('########## Begin Evaluation ############' + '\n')
            f.write(json.dumps(youcook_caption_log) + '\n')
            f.write(json.dumps(ego4dcap_log) + '\n')
            f.write('########## Done Evaluation ############' + '\n')
            
    ### save train stats ###
    train_stats_dict = {f'train_{k}': round(v, 3) for k, v in train_stats.items()}
    val_stats_dict = {}
    if (epoch + 1) % args.eval_freq == 0:
        # TODO: add evaluation
        val_stats = validate(val_loader, model, criterion, args)
        val_stats_dict = {f'test_{k}': round(v, 3) for k, v in val_stats.items()}
    
    log_stats = {**train_stats_dict, **val_stats_dict}

    if dist_utils.is_main_process():
        if args.wandb:
            wandb.log(log_stats)
        with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
            f.write(json.dumps(log_stats) + '\n')
        
    
def data_visualization(train_loader):    
    inputs = next(iter(train_loader)) # dict
    caption = inputs['text'] #[b, 77]
    videos = inputs['video'] #[b, c, t, h, w]
    uids = inputs['uid']
    start_seconds = inputs['start_second']
    end_seconds = inputs['end_second']
    anno_uids = inputs['anno_uid']

    for bs in range(len(caption)):
        sample_caption = caption[bs] #[77]
        sample_video = videos[bs] #[c, t, h, w]

        endloc = torch.where(sample_caption == 49407)[0]
        raw_caption = train_loader.dataset.tokenizer.decode(sample_caption[1:endloc].numpy())
       
        print(bs, raw_caption)
        print(uids[bs], anno_uids[bs])
        print(start_seconds[bs], end_seconds[bs])
        print()

        video = sample_video.permute(1, 0, 2, 3) #[t, 3, 224, 224]
        for i in range(video.shape[0]):
            image = video[i].cpu().numpy() #[224, 224, 3]  
            # image = (image * 255).astype(np.uint8)
            mean = [122.77, 116.7460125, 104.09373615000001]
            std =[68.5005327, 66.6321579, 70.32316305]
            image[0] = image[0] * std[0] + mean[0]
            image[1] = image[1] * std[1] + mean[1]
            image[2] = image[2] * std[2] + mean[2]
            
            image = image.transpose(1, 2, 0).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'vis/video_{bs}_frame_{i}.jpg', image)


def interpolate_clip_positional_embeds(new_length, pos_embed):
    '''
    # positional_embed: [self.ctx_length, D]
    # 
    # '''
    old_length = pos_embed.data.shape[0]
    if new_length == old_length:
        return pos_embed
    if new_length < old_length:
        return pos_embed[:new_length, :]

    new_temporal_embed = F.interpolate(pos_embed.unsqueeze(0).unsqueeze(0), (new_length, pos_embed.shape[-1]), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
    return new_temporal_embed
