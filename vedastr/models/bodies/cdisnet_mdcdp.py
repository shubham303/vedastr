'''
Description: CDistNet
version: 1.0
Author: YangBitao
Date: 2021-11-26 17:36:54
Contact: yangbitao001@ke.com
'''

# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))

from vedastr.models.bodies import build_sequence_encoder
from vedastr.models.bodies.registry import BODIES
from vedastr.models.utils import build_torch_nn

sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@BODIES.register_module
class MDCDP(nn.Module):
    def __init__(self, sae_config, cbi_s_config, cbi_v_config,d_model,activation, pos_mask, vis_mask, sem_mask,
                 vis_mask_range , sem_mask_range):
        
        super(MDCDP, self).__init__()
        
        self.sae = build_sequence_encoder(sae_config)
        self.cbi_v = build_sequence_encoder(cbi_s_config)
        self.cbi_s = build_sequence_encoder(cbi_v_config)
        
        self.conv = nn.Conv2d(d_model * 2, d_model, kernel_size=1,
                                  stride=1, padding=0, bias=False)
        
        self.active = build_torch_nn(activation)

        self.pos_mask= pos_mask
        self.vis_mask=vis_mask
        self.sem_mask = sem_mask
        self.vis_mask_range= vis_mask_range
        self.sem_mask_range = sem_mask_range
        self.mask_cached = False
        
    def get_masks(self, pos_embedding, vis_feature, sem_embedding):
        """
        no need to compute mask in every iteration.
        """
        if not self.mask_cached :
            self.pos_mask = self.pos_mask(pos_embedding.size(1), pos_embedding.size(1), self.sem_mask_range).to(device)
            self.vis_mask = self.vis_mask(pos_embedding.size(1), vis_feature.size(1), self.vis_mask_range).to(device)
            self.sem_mask = self.sem_mask(pos_embedding.size(1), sem_embedding.size(1), self.sem_mask_range).to(device)
            self.mask_cached=True
        
        return self.pos_mask, self.vis_mask, self.sem_mask
        
    def forward(self, pos_embedding, vis_feature, sem_embedding):
        # self attention enhancement
        pos_mask, vis_mask, sem_mask = self.get_masks(pos_embedding, vis_feature, sem_embedding)
        pos_feature = self.sae(pos_embedding, pos_embedding, pos_embedding, pos_mask)
        vis_feature = self.cbi_v(pos_feature, vis_feature, vis_feature, vis_mask)
        sem_embedding = self.cbi_s(pos_feature, sem_embedding, sem_embedding, sem_mask)
        

        context = torch.cat([vis_feature, sem_embedding], dim=2)
        context = context.permute(0, 2, 1).unsqueeze(2)
        context = self.conv(context)
        context = context.squeeze(2).permute(0, 2, 1)
        context = self.active(context)
        context_vis_feature = (1 - context) * vis_feature
        context_sem_embedding = context * sem_embedding
        fuse_feature = context_vis_feature + context_sem_embedding

        return fuse_feature
    
        

    
