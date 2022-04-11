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
import bisect
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))


from vedastr.models.bodies import build_body, build_sequence_decoder
from .registry import MODELS


sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '/')))

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



@MODELS.register_module
class Cdisnet(nn.Module):
    def __init__(self, vis_module, sem_module, pos_module, mdcdp_layers,share_weight, d_model,num_class,
                 max_seq_len,  need_text,need_lang , language_embedding):
        
        super(Cdisnet, self).__init__()
        self.need_text=need_text
        self.need_lang = need_lang   #set to True, if recognition is multilingual and requires language id for
        # prediction. set To false if recognition models itself identifies script( lang) of the text in the image
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
        self.language_embedding = build_body(language_embedding)   # language_embedding translates language id to
        # feature vector
    
            
    def forward(self, inputs , beam_size =0):
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

       
        if language is not None:                         # language embedding set None means model is single languge
        # model
            
            language_embedding = self.language_embedding(language)
            language_embedding = language_embedding.unsqueeze(1)
            vis_feature[:] = vis_feature+language_embedding              # add language embedding to each vis vector
        

        # classifier
        if self.training:
            outputs = self.process(input_char, pos_embedding, vis_feature)
            outputs = self.linear(outputs)
            return outputs
        else:
            if not beam_size or beam_size <=0:
                
                outputs = []
                for i in range(self.max_seq_len):
                    fuse_feature = self.process(input_char, pos_embedding, vis_feature)
                    fuse_feature_step = fuse_feature[:, i, :]
                    fuse_feature_step = self.linear(fuse_feature_step)
                    outputs.append(fuse_feature_step)
                    
                    fuse_feature_step = F.softmax(fuse_feature_step, dim=-1)
                
                    _, max_idx = torch.max(fuse_feature_step, dim=1, keepdim=False)
        
                    if i < self.max_seq_len - 1:
                        input_char[:, i+1] = max_idx
                
                        
                outputs = torch.stack(outputs, dim=1)
                return outputs
            
            else:
                outputs = []
                outputs_indexes= []
                batch_size = input_char.size(0)
                
                for b_idx in range(0, batch_size):
                    sequences = [(0.0, list(), input_char[b_idx].unsqueeze(0), list())]  # score, outputs, input_char,
                    for i in range(self.max_seq_len):
                        X=[]
                        for j in range(len(sequences)):
                            score, b_outputs, b_input_char, indexes = sequences.pop(0)  # remove first element
                            fuse_feature = self.process(b_input_char, pos_embedding[b_idx].unsqueeze(0),
                                                        vis_feature[b_idx].unsqueeze(0))
                            fuse_feature_step = fuse_feature[:, i, :]
                            fuse_feature_step = self.linear(fuse_feature_step)
                            fuse_feature_step_softmax = F.softmax(fuse_feature_step, dim=-1)
                            top_k_features, top_k_idx = torch.topk(fuse_feature_step_softmax, dim=-1, k=beam_size,
                                                                   largest=True)
                            for k in range(0, beam_size):
                                b_outputs_k = b_outputs.copy()
                                b_input_char_k = torch.clone(b_input_char)
                                indexes_k = indexes.copy()
                                b_outputs_k.append(fuse_feature_step)
    
                                val, idx = top_k_features[0, k], top_k_idx[0, k]
                                indexes_k.append(idx)
                                if i < self.max_seq_len - 1:
                                    b_input_char_k[:, i + 1] = idx
                                score -= torch.log(val).item()
    
                      
                                bisect.insort(X, (score, b_outputs_k, b_input_char_k, indexes_k))
                                X = X[:min(beam_size, len(X))]
                        sequences = X

                        outputs.append(torch.stack(sequences[0][1], dim=1))
                        outputs_indexes.append(torch.stack(sequences[0][3], dim=0))

                    outputs = torch.stack(outputs, dim=0).squeeze(1)
                    outputs_indexes = torch.stack(outputs_indexes, dim=0)
                    return outputs, outputs_indexes
        

    def process(self, input_char, pos_embedding, vis_feature):
        sem_embedding = self.sem_module(input_char)
        outputs = self.mdcdp_layers[0](pos_embedding, vis_feature, sem_embedding)
        for i in range(1, len(self.mdcdp_layers)):
            outputs = self.mdcdp_layers[i](outputs, vis_feature, sem_embedding)
        return outputs


if __name__ == "__main__":
    input = torch.randn(2, 3, 128, 32).to(device)
    input_char = torch.randint(100, size=(2, 20)).to(device)
    cdistnet = Cdisnet().to(device)
    output = cdistnet(input, input_char)
    print(output.shape)








