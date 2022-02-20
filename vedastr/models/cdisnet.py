'''
Description: CDistNet
version: 1.0
Author: YangBitao
Date: 2021-11-26 17:33:42
Contact: yangbitao001@ke.com
'''

# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))

import abfn.abfn

from vedastr.models.bodies import build_body, build_sequence_decoder
from vedastr.models.bodies.sequences.transformer.embedding.builder import build_embedding_layer
from .registry import MODELS
from einops import rearrange


sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '/')))

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



@MODELS.register_module
class Cdisnet(nn.Module):
    def __init__(self, vis_module, sem_module, pos_module, mdcdp_layers,share_weight, d_model,num_class,  max_seq_len,  need_text,language_embedding):
        
        super(Cdisnet, self).__init__()
        self.need_text=need_text
        self.vis_module = build_body(vis_module)
        self.pos_module =build_body(pos_module)
        self.sem_module = build_body(sem_module)
        
        if share_weight:
            mdcdp_layer= build_body(mdcdp_layers[0])
            self.mdcdp_layers  = nn.ModuleList([mdcdp_layer]*len(mdcdp_layers))
        else:
            self.mdcdp_layers = nn.ModuleList([build_body(mdcdp) for mdcdp in mdcdp_layers])
        
        self.linear = nn.Linear(d_model,num_class)
        self.max_seq_len = max_seq_len
        

        self.language_embedding = build_body(language_embedding)
           
            
    def forward(self, inputs):
        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]
            
        input = inputs[0]
        input_char = inputs[1]
        language=   inputs[2]  if len(inputs) >2 else None
        
        if not self.training:
            batch_size = input.size(0)
            input_char = torch.full((batch_size, self.max_seq_len), torch.clone(input_char[0, 0])).long().to(
                device)
            
        
        vis_feature = self.vis_module(input)
        pos = input_char.new_zeros(*input_char.shape).unsqueeze(2)
        pos_embedding = self.pos_module(pos)

       
        language_embedding = self.language_embedding(language)
        language_embedding = language_embedding.unsqueeze(1)
        vis_feature[:] = vis_feature+language_embedding
            
        if self.training:
            sem_embedding = self.sem_module(input_char)
            outputs = self.mdcdp_layers[0](pos_embedding, vis_feature, sem_embedding)
            
            for i in range(1, len(self.mdcdp_layers)):
                outputs=self.mdcdp_layers[i](outputs, vis_feature, sem_embedding)
                
            outputs = self.linear(outputs)
        else:
            outputs = []
            for i in range(self.max_seq_len):
                sem_embedding = self.sem_module(input_char)
                fuse_feature = self.mdcdp_layers[0](pos_embedding, vis_feature, sem_embedding)
                
                for l in range(1, len(self.mdcdp_layers)):
                    fuse_feature = self.mdcdp_layers[l](fuse_feature, vis_feature, sem_embedding)
                
                fuse_feature_step = fuse_feature[:, i, :]
                fuse_feature_step = self.linear(fuse_feature_step)
                outputs.append(fuse_feature_step)
                
                fuse_feature_step = F.softmax(fuse_feature_step, dim=-1)
                #fuse_feature_step = fuse_feature_step.view(64, -1)
                _, max_idx = torch.max(fuse_feature_step, dim=1, keepdim=False)
                #input_char = input_char.repeat(3, 1, 1).transpose(1,0)
                
               # vis_feature = vis_feature.repeat(3,1,1)
                #pos_embedding = pos_embedding.repeat(3, 1, 1)
                if i < self.max_seq_len - 1:
                    input_char[:, i+1] = max_idx
                    #input_char = rearrange(input_char, 'd0 d1 d2 -> (d0 d1) d2')
                    
            outputs = torch.stack(outputs, dim=1)

        return outputs




if __name__ == "__main__":
    input = torch.randn(2, 3, 128, 32).to(device)
    input_char = torch.randint(100, size=(2, 20)).to(device)
    cdistnet = CDistNet().to(device)
    output = cdistnet(input, input_char)
    print(output.shape)








