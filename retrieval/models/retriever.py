from .openai_model import QuickGELU, Transformer
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import io
from function.func_utils import interpolate_clip_positional_embeds

class CLIP_FEAT(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 vision_length: int,
                 vision_transformer_width: int,
                 vision_transformer_heads: int,
                 vision_transformer_layers: int,
                 vision_encoder,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 tempearture_init=0.07,
                 **kwargs,
                 ):
        super().__init__()

        self.context_length = context_length
        self.vision_transformer_width = vision_transformer_width
        self.vision_length = vision_length

        if vision_encoder is None:
            self.bert_encoder = False
            self.visual = Transformer(
                width=vision_transformer_width,
                layers=vision_transformer_layers,
                heads=vision_transformer_heads,
            )
            self.visual_positional_embedding = self.sinusoidal_positional_embedding(vision_length, vision_transformer_width)
            self.visual_ln_final = nn.LayerNorm(vision_transformer_width)
        else:
            self.bert_encoder = True
            ### default has [512, 768] pos embed 
            self.visual = vision_encoder
            # self.visual.embedding.
        
        self.attn_mask = self.build_attention_mask()
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.attn_mask,
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)

        # self.visual_positional_embedding = nn.Parameter(torch.empty(self.vision_length, vision_transformer_width))
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = nn.LayerNorm(transformer_width)  # used to be `models.transformer.LayerNorm``

        self.aggregation_projection = nn.Parameter(torch.empty(vision_transformer_width, embed_dim))
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / tempearture_init))
        print("=> initialize initial temperature with {} and logit scale {}".format(tempearture_init, self.logit_scale))
        
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

        nn.init.normal_(self.aggregation_projection, std=self.vision_transformer_width ** -0.5)
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
    
    def sinusoidal_positional_embedding(self, T=4, d=768, n=10000.0):
        # T, d = self.vision_length, self.vision_transformer_width
        if d % 2 != 0:
            raise ValueError("Sinusoidal positional embedding cannot apply to odd token embedding dim (got dim={:d})".format(d))

        positions = torch.arange(0, T).unsqueeze_(1)
        embeddings = torch.zeros(T, d)

        denominators = torch.pow(n, 2*torch.arange(0, d//2)/d) # 10000^(2i/d_model), i is the index of embedding
        embeddings[:, 0::2] = torch.sin(positions/denominators) # sin(pos/10000^(2i/d_model))
        embeddings[:, 1::2] = torch.cos(positions/denominators) # cos(pos/10000^(2i/d_model))
        return embeddings

    def inflate_temporal_embeds(self, curr_frames):
        if self.training is False and self.vision_length != curr_frames:
            return self.sinusoidal_positional_embedding(curr_frames, self.vision_transformer_width)
        return self.visual_positional_embedding

    def inflate_positional_embeds(self, curr_frames):
        '''
        # positional_embed: [self.ctx_length, D]
        # 
        # '''
        if self.context_length == curr_frames:
            return self.positional_embedding, self.attn_mask
        if self.context_length > curr_frames:
            # return self.positional_embedding[:, :curr_frames, :], self.build_attention_mask(curr_frames)
            return self.positional_embedding[:curr_frames, :], self.build_attention_mask(curr_frames)
        if self.context_length < curr_frames:
            new_temporal_embed = F.interpolate(self.positional_embedding.unsqueeze(0).unsqueeze(0), (curr_frames, self.positional_embedding.shape[-1]), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            return torch.nn.Parameter(new_temporal_embed).to(self.positional_embedding.device), self.build_attention_mask(curr_frames)

    def encode_image(self, image, use_checkpoint=False, apply_project=True):
        # x = image
        if self.bert_encoder:
            x = self.visual(inputs_embeds=image)
            x = x.last_hidden_state
        else:
            # print(image.shape)
            # print(self.visual_positional_embedding.to(image.device).unsqueeze(0).shape)
            # st()
            curr_frames = image.size(1)
            visual_positional_embedding = self.inflate_temporal_embeds(curr_frames)
            x = image + visual_positional_embedding.to(image.device).unsqueeze(0)
            x = x.permute(1, 0, 2)
            # x = self.visual(x, use_checkpoint=use_checkpoint)
            x = self.visual(x)
            x = x.permute(1, 0, 2)
            x = self.visual_ln_final(x)
            if not apply_project:
                return x
            
        x = x.mean(1)
        if self.aggregation_projection is not None:
            x = x @ self.aggregation_projection
            
        return x

    def encode_text(self, text, use_checkpoint=False):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        curr_ctx_len = x.shape[1]
        positional_embedding, attn_mask = self.inflate_positional_embeds(curr_ctx_len)
        # print(positional_embedding.shape)
        # st()
        x = x + positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, use_checkpoint=use_checkpoint, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text, use_checkpoint=False, norm_embed=False):
        image_embed = self.encode_image(image, use_checkpoint=use_checkpoint)
        text_embed = self.encode_text(text, use_checkpoint=use_checkpoint)

        if norm_embed:
            image_embed = F.normalize(image_embed, dim=-1)
            text_embed = F.normalize(text_embed, dim=-1)
        return {'image_embed': image_embed,
                'text_embed': text_embed,
                'logit_scale': self.logit_scale.exp()}


def CLIP_FEAT_LARGE_OPENAI(
    num_frames=4, temperature_init=0.07, project_embed_dim=768, freeze_text_encoder=False, 
    vision_transformer_width=768, text_transformer_width=768, context_length=77,
    **kwargs,
):
    clip_model, _ = clip.load('ViT-L/14', 'cpu')
    clip_text_width = 768
    
    vision_encoder = None
    model = CLIP_FEAT(
        embed_dim=project_embed_dim,
        vision_length=num_frames,
        vision_encoder=vision_encoder,
        vision_transformer_width=vision_transformer_width,
        vision_transformer_heads=8,
        vision_transformer_layers=4,
        context_length=context_length,
        vocab_size=49408,
        transformer_width=text_transformer_width,
        transformer_heads=8,
        transformer_layers=12,
        tempearture_init=temperature_init,
        **kwargs
    )
    if text_transformer_width == clip_text_width:
        model.transformer.load_state_dict(clip_model.transformer.state_dict())
        model.token_embedding.load_state_dict(clip_model.token_embedding.state_dict())
        ######
        # model.positional_embedding.data.copy_(clip_model.positional_embedding.data)
        ## handle larger context length
        interpolated_pos_embed = interpolate_clip_positional_embeds(context_length, clip_model.positional_embedding)
        model.positional_embedding.data.copy_(interpolated_pos_embed.data)
        print(f'=> Interpolate the positional embedding shape from {clip_model.positional_embedding.data.shape[0]} to {context_length}')
        #######
        model.ln_final.load_state_dict(clip_model.ln_final.state_dict())
        if project_embed_dim == clip_model.text_projection.shape[1]:
            print("=> Loading CLIP's text_projection, image_projection and logit_scale directly")
        #st()
        # model.image_projection.data.copy_(clip_model.visual.proj.data)
        # model.text_projection.data.copy_(clip_model.text_projection.data)
        # model.logit_scale.data.copy_(clip_model.logit_scale.data)
    
    if freeze_text_encoder:
        for module in [model.token_embedding, model.positional_embedding, model.transformer, model.ln_final, model.text_projection]:
            if isinstance(module, nn.Parameter):
                module.requires_grad = False
            else:
                for p in module.parameters():
                    p.requires_grad=False
            
    return model
