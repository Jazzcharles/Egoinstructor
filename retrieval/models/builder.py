
from mmcv.utils import Registry
from omegaconf import OmegaConf
from models import retriever
import torch

def build_model(config):
    if 'FEAT' in config.name:
        model = getattr(retriever, config.name)(
            num_frames=config.num_frames,
            project_embed_dim=config.project_embed_dim,
            temperature_init=config.temperature_init,
            freeze_text_encoder=config.freeze_text_encoder,
            vision_transformer_width=config.vision_transformer_width,
            text_transformer_width=config.text_transformer_width,
            context_length=config.ctx_length,
        )
    else:
        raise NotImplementedError

    return model