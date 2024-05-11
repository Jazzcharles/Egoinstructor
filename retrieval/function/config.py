
import os
import os.path as osp

from omegaconf import OmegaConf


def load_config(cfg_file):
    cfg = OmegaConf.load(cfg_file)
    if '_base_' in cfg:
        if isinstance(cfg._base_, str):
            base_cfg = OmegaConf.load(osp.join(osp.dirname(cfg_file), cfg._base_))
        else:
            base_cfg = OmegaConf.merge(OmegaConf.load(f) for f in cfg._base_)
        cfg = OmegaConf.merge(base_cfg, cfg)
    return cfg

def get_config(args):
    cfg = load_config(args.config)
    OmegaConf.set_struct(cfg, True)

    # if args.opts is not None:
    #     cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.opts))
    # if hasattr(args, 'batch_size') and args.batch_size:
    #     cfg.train.batch_size = args.batch_size

    # if hasattr(args, 'resume') and args.resume:
    #     cfg.resume = args.resume

    # if hasattr(args, 'testonly') and args.testonly:
    #     cfg.test.testonly = args.testonly

    if hasattr(args, 'output') and args.output:
        cfg.output = args.output
   
    if hasattr(args, 'wandb') and args.wandb:
        cfg.wandb = args.wandb

    cfg.local_rank = args.local_rank

    OmegaConf.set_readonly(cfg, True)

    return cfg
