# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import torch
import torch.distributed as dist
import torch.distributed.nn
import torch.nn as nn
import torch.nn.functional as F

from function.func_utils import gather_from_all

class EgoExoNCE(nn.Module):

    def __init__(
            self,
            cfg=None,
            world_size=1,
    ):
        super().__init__()
        self.cfg=cfg
        self.world_size = world_size

        assert self.cfg is not None
        # self.topK = cfg.data.topk + 1 ### important here, add the current sample ###
        self.topK = cfg.data.pair_num + 1
        self.total_bs = self.world_size * self.cfg.train.batch_size * self.topK

        ### for fast compute cross-view masks ###
        chunks = self.total_bs // self.topK
        self.col_indices, self.row_indices = [], []
        for t in range(0, chunks):
            self.col_indices.append(torch.arange(t * self.topK, (t + 1) * self.topK).repeat(self.topK))
            self.row_indices.append(torch.arange(t * self.topK, (t + 1) * self.topK).repeat_interleave(self.topK))
        self.col_indices = torch.stack(self.col_indices).flatten()
        self.row_indices = torch.stack(self.row_indices).flatten()


    def forward(self, outputs):
        loss = {}
        
        ### get logits ###
        image_features, text_features = outputs['image_embed'], outputs['text_embed']
        logit_scale = outputs['logit_scale']
        all_image_features = gather_from_all(image_features)
        all_text_features = gather_from_all(text_features)
        logits_per_image = logit_scale * all_image_features @ all_text_features.T
        logits_per_text = logits_per_image.T
        
        total_bs = all_image_features.shape[0]
        device = image_features.device
        
        ### initial mask from the main diag ###
        mask_diag = torch.eye(total_bs).cuda()
        mask = mask_diag
        
        ### Calculate the batch-wise similarity to select samples with same noun/verb ###
        noun_vec, verb_vec = outputs['noun_vec'], outputs['verb_vec']
        all_nouns = gather_from_all(noun_vec)
        all_verbs = gather_from_all(verb_vec)
        sim_nouns = sim_matrix(all_nouns, all_nouns)
        sim_verbs = sim_matrix(all_verbs, all_verbs)
        mask = mask + sim_nouns * sim_verbs
        
        ### Calculate the mask for cross-view pairs ###
        mask_sim = torch.zeros(total_bs, total_bs).to(device)
        if total_bs != self.total_bs:
            chunks = total_bs // self.topK
            col_indices, row_indices = [], []
            for t in range(0, chunks):
                col_indices.append(torch.arange(t * self.topK, (t + 1) * self.topK).repeat(self.topK))
                row_indices.append(torch.arange(t * self.topK, (t + 1) * self.topK).repeat_interleave(self.topK))
            col_indices = torch.stack(col_indices).to(device).flatten()
            row_indices = torch.stack(row_indices).to(device).flatten()
        else:
            col_indices = self.col_indices.to(device)
            row_indices = self.row_indices.to(device)
            
        mask_sim[row_indices, col_indices] += 1
        mask = mask + mask_sim
        
        ### Calculate the contrastive loss ###        
        i_sm = F.softmax(logits_per_image, dim=1)
        j_sm = F.softmax(logits_per_text, dim=1)

        mask_bool = mask > 0
        idiag = torch.log(torch.sum(i_sm * mask_bool, dim=1) )
        loss_i = idiag.sum() / len(idiag)
        
        jdiag = torch.log(torch.sum(j_sm * mask_bool, dim=1) )
        loss_j = jdiag.sum() / len(jdiag)
        
        total_loss = - loss_i - loss_j
        return {
            'loss': total_loss,
            'clip_loss': total_loss,
            'clip_acc': total_loss,
        }

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt
