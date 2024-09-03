from .openai_model import QuickGELU, Transformer
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import io
from models.videomae.videomaev2 import PretrainVisionTransformerEncoder

def FLIP_VideoMAEV2_LARGE_OPENAI(
    num_frames=8, img_size=224, patch_size=16, encoder_in_chans=3, encoder_num_classes=0, encoder_embed_dim=1024, encoder_depth=24, encoder_num_heads=16, 
    mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,norm_layer=nn.LayerNorm,
    init_values=0., use_learnable_pos_emb=False, tubelet_size=2, num_classes=0,  # avoid the error from create_fn in timm
    in_chans=0,  # avoid the error from create_fn in timm
    with_cp=False, cos_attn=False,
    pretrained_visual_checkpoint=None,
    clip_visual_teacher=False,
    temperature_init=0.07, project_embed_dim=256, freeze_text_encoder=False, **kwargs,
):
    vision_model = PretrainVisionTransformerEncoder(
        img_size=img_size, patch_size=patch_size, in_chans=encoder_in_chans, num_classes=encoder_num_classes,
        embed_dim=encoder_embed_dim, depth=encoder_depth, num_heads=encoder_num_heads,
        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
        norm_layer=norm_layer, init_values=init_values, tubelet_size=tubelet_size, use_learnable_pos_emb=use_learnable_pos_emb,
        with_cp=with_cp, all_frames=num_frames, cos_attn=cos_attn,
    )
    
    clip_model, _ = clip.load('ViT-L/14', 'cpu')
    if pretrained_visual_checkpoint is not None:
        print("=> Loading Pre-trained VideoMAE weights")    
        visual_ckpt = torch.load(pretrained_visual_checkpoint, map_location='cpu')
        visual_ckpt = visual_ckpt['module'] if 'module' in visual_ckpt else visual_ckpt
        res = vision_model.load_state_dict(visual_ckpt, strict=False)
        print(res)

    vision_model.head = nn.Identity()
    vision_model.pre_logits = nn.Identity()
    vision_model.fc = nn.Identity()
    model = FLIP(
        embed_dim=project_embed_dim,
        vision_width=encoder_embed_dim,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=768,
        transformer_heads=8,
        transformer_layers=12,
        tempearture_init=temperature_init,
        **kwargs
    )
    model.transformer.load_state_dict(clip_model.transformer.state_dict())
    model.token_embedding.load_state_dict(clip_model.token_embedding.state_dict())
    model.positional_embedding.data.copy_(clip_model.positional_embedding.data)
    model.ln_final.load_state_dict(clip_model.ln_final.state_dict())
    if project_embed_dim == clip_model.text_projection.shape[1]:
        print("=> Loading CLIP's text_projection, image_projection and logit_scale directly")
        ### for videoMAE, we have no video proj ###
        # model.image_projection.data.copy_(clip_model.visual.proj.data)
        model.text_projection.data.copy_(clip_model.text_projection.data)
        model.logit_scale.data.copy_(clip_model.logit_scale.data)
    
    if freeze_text_encoder:
        for module in [model.token_embedding, model.positional_embedding, model.transformer, model.ln_final, model.text_projection]:
            if isinstance(module, nn.Parameter):
                module.requires_grad = False
            else:
                for p in module.parameters():
                    p.requires_grad=False
            
    return model


class FLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 vision_width: int,
                 vision_model: nn.Module,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 tempearture_init=0.07,
                 clip_visual_teacher=None,
                 **kwargs,
                 ):
        super().__init__()

        self.context_length = context_length
        self.vision_width = vision_width

        self.visual = vision_model
        self.attn_mask = self.build_attention_mask()
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.attn_mask,
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = nn.LayerNorm(transformer_width)  # used to be `models.transformer.LayerNorm``

        self.image_projection = nn.Parameter(torch.empty(vision_width, embed_dim))
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        print("=> initialize initial temperature with {}".format(tempearture_init))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / tempearture_init))
        self.clip_visual_teacher = clip_visual_teacher

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self, ctx_length=None):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        if ctx_length is None:
            mask = torch.empty(self.context_length, self.context_length)
        else:
            mask = torch.empty(ctx_length, ctx_length)

        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def inflate_positional_embeds(self, curr_frames):
        '''
        # positional_embed: [self.ctx_length, D]
        # 
        # '''
        if self.context_length == curr_frames:
            return self.positional_embedding, self.attn_mask
        if self.context_length > curr_frames:
            return self.positional_embedding[:, :curr_frames, :], self.build_attention_mask(curr_frames)
        if self.context_length < curr_frames:
            new_temporal_embed = F.interpolate(self.positional_embedding.unsqueeze(0).unsqueeze(0), (curr_frames, self.positional_embedding.shape[-1]), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            return torch.nn.Parameter(new_temporal_embed).to(self.positional_embedding.device), self.build_attention_mask(curr_frames)

    def encode_image(self, image, encoder_mask=None, use_checkpoint=False, apply_project=True):
        x = self.visual(image, encoder_mask)

        if isinstance(x, list):
            assert len(x) == 1
            x = x[0]
            
        ### apply mean over image tokens ###
        if len(x.size()) == 3:
            x = x.mean(1)
        if not apply_project:
            return x
        x = x @ self.image_projection
        return x

    def encode_text(self, text, use_checkpoint=False):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        # x = x + self.positional_embedding
        curr_ctx_len = x.shape[1]
        positional_embedding, attn_mask = self.inflate_positional_embeds(curr_ctx_len)
        x = x + positional_embedding

        x = x.permute(1, 0, 2)  # NLD -> LND
        # x = self.transformer(x, use_checkpoint=use_checkpoint)
        x = self.transformer(x, use_checkpoint=use_checkpoint, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text, encoder_mask=None, use_checkpoint=False, norm_embed=False):
        image_embed = self.encode_image(image, encoder_mask, use_checkpoint=use_checkpoint)
        text_embed = self.encode_text(text, use_checkpoint=use_checkpoint)

        if norm_embed:
            image_embed = F.normalize(image_embed, dim=-1)
            text_embed = F.normalize(text_embed, dim=-1)
        return {'image_embed': image_embed,
                'text_embed': text_embed,
                'logit_scale': self.logit_scale.exp()}