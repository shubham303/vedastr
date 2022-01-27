'''
Description: CDistNet
version: 1.0
Author: YangBitao
Date: 2021-11-26 15:24:32
Contact: yangbitao001@ke.com
'''


# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))

from vedastr.models.bodies import build_component, resnet
from vedastr.models.bodies.registry import BODIES
import torch.nn.functional as F

sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import torch
import torch.nn as nn

from transformer import TransformerUnit

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@BODIES.register_module
class VisualModule(nn.Module):
    def __init__(self, tps , d_input, layers, n_layer, d_model, d_inner,
                    n_head, d_k, d_v, dropout=0.1):
        super(VisualModule, self).__init__()
        self.tps =build_component(tps)
        self.feature_extractor = Feature_Extractor([(1,1), (2,2), (1,1), (2,2), (1,1), (1,1)], False, [1, 32, 128])
        self.CAM= CAM(self.feature_extractor.Iwantshapes(), 25, 8, 64)
        
    def forward(self, x):
        x= self.tps(x)
        x= self.feature_extractor(x)
        A= self.CAM(x)

        nB, nC, nH, nW = x[-1].size()
        nT = A.size()[1]
        # Normalize
        A = A / A.view(nB, nT, -1).sum(2).view(nB, nT, 1, 1)
        # weighted sum
        C = x[-1].view(nB, 1, nC, nH, nW) * A.view(nB, nT, 1, nH, nW)
        C = C.view(nB, nT, nC, -1).sum(3)
        return C
    
    
class Feature_Extractor(nn.Module):
    def __init__(self, strides, compress_layer, input_shape):
        super(Feature_Extractor, self).__init__()
        self.model = resnet.resnet45(strides, compress_layer)
        self.input_shape = input_shape

    def forward(self, input):
        features = self.model(input)
        return features

    def Iwantshapes(self):
        pseudo_input = torch.rand(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        features = self.model(pseudo_input)
        return [feat.size()[1:] for feat in features]


class CAM(nn.Module):
    def __init__(self, scales, maxT, depth, num_channels):
        super(CAM, self).__init__()
        # cascade multiscale features
        fpn = []
        for i in range(1, len(scales)):
            assert not (scales[i-1][1] / scales[i][1]) % 1, 'layers scale error, from {} to {}'.format(i-1, i)
            assert not (scales[i-1][2] / scales[i][2]) % 1, 'layers scale error, from {} to {}'.format(i-1, i)
            ksize = [3,3,5] # if downsampling ratio >= 3, the kernel size is 5, else 3
            r_h, r_w = int(scales[i-1][1] / scales[i][1]), int(scales[i-1][2] / scales[i][2])
            ksize_h = 1 if scales[i-1][1] == 1 else ksize[r_h-1]
            ksize_w = 1 if scales[i-1][2] == 1 else ksize[r_w-1]
            fpn.append(nn.Sequential(nn.Conv2d(scales[i-1][0], scales[i][0],
                                              (ksize_h, ksize_w),
                                              (r_h, r_w),
                                              (int((ksize_h - 1)/2), int((ksize_w - 1)/2))),
                                     nn.BatchNorm2d(scales[i][0]),
                                     nn.ReLU(True)))
        self.fpn = nn.Sequential(*fpn)
        # convolutional alignment
        # convs
        assert depth % 2 == 0, 'the depth of CAM must be a even number.'
        in_shape = scales[-1]
        strides = []
        conv_ksizes = []
        deconv_ksizes = []
        h, w = in_shape[1], in_shape[2]
        for i in range(0, int(depth / 2)):
            stride = [2] if 2 ** (depth/2 - i) <= h else [1]
            stride = stride + [2] if 2 ** (depth/2 - i) <= w else stride + [1]
            strides.append(stride)
            conv_ksizes.append([3, 3])
            deconv_ksizes.append([_ ** 2 for _ in stride])
        convs = [nn.Sequential(nn.Conv2d(in_shape[0], num_channels,
                                        tuple(conv_ksizes[0]),
                                        tuple(strides[0]),
                                        (int((conv_ksizes[0][0] - 1)/2), int((conv_ksizes[0][1] - 1)/2))),
                               nn.BatchNorm2d(num_channels),
                               nn.ReLU(True))]
        for i in range(1, int(depth / 2)):
            convs.append(nn.Sequential(nn.Conv2d(num_channels, num_channels,
                                                tuple(conv_ksizes[i]),
                                                tuple(strides[i]),
                                                (int((conv_ksizes[i][0] - 1)/2), int((conv_ksizes[i][1] - 1)/2))),
                                       nn.BatchNorm2d(num_channels),
                                       nn.ReLU(True)))
        self.convs = nn.Sequential(*convs)
        # deconvs
        deconvs = []
        
        for i in range(1, int(depth / 2)):
            deconvs.append(nn.Sequential(nn.ConvTranspose2d(num_channels, num_channels,
                                                           tuple(deconv_ksizes[int(depth/2)-i]),
                                                           tuple(strides[int(depth/2)-i]),
                                                           (int(deconv_ksizes[int(depth/2)-i][0]/4.), int(deconv_ksizes[int(depth/2)-i][1]/4.))),
                                         nn.BatchNorm2d(num_channels),
                                         nn.ReLU(True)))
        deconvs.append(nn.Sequential(nn.ConvTranspose2d(num_channels, maxT,
                                                       tuple(deconv_ksizes[0]),
                                                       tuple(strides[0]),
                                                       (int(deconv_ksizes[0][0]/4.), int(deconv_ksizes[0][1]/4.))),
                                     nn.Sigmoid()))
        self.deconvs = nn.Sequential(*deconvs)
    def forward(self, input):
        x = input[0]
        for i in range(0, len(self.fpn)):
            x = self.fpn[i](x) + input[i+1]
        conv_feats = []
        for i in range(0, len(self.convs)):
            x = self.convs[i](x)
            conv_feats.append(x)
        for i in range(0, len(self.deconvs) - 1):
            x = self.deconvs[i](x)
            x = x + conv_feats[len(conv_feats) - 2 - i]
        x = self.deconvs[-1](x)
        return x



class DTD(nn.Module):
    # LSTM DTD
    def __init__(self, nclass, nchannel, dropout = 0.3):
        super(DTD,self).__init__()
        self.nclass = nclass
        self.nchannel = nchannel
        self.pre_lstm = nn.LSTM(nchannel, int(nchannel / 2), bidirectional=True)
        self.rnn = nn.GRUCell(nchannel * 2, nchannel)
        self.generator = nn.Sequential(
                            nn.Dropout(p = dropout),
                            nn.Linear(nchannel, nclass)
                        )
        self.char_embeddings = nn.Parameter(torch.randn(nclass, nchannel))

    def forward(self, feature, A, text, text_length, test = False):
        nB, nC, nH, nW = feature.size()
        nT = A.size()[1]
        # Normalize
        A = A / A.view(nB, nT, -1).sum(2).view(nB,nT,1,1)
        # weighted sum
        C = feature.view(nB, 1, nC, nH, nW) * A.view(nB, nT, 1, nH, nW)
        C = C.view(nB,nT,nC,-1).sum(3).transpose(1,0)
        C, _ = self.pre_lstm(C)
        C = F.dropout(C, p = 0.3, training=self.training)
        if not test:
            lenText = int(text_length.sum())
            nsteps = int(text_length.max())

            gru_res = torch.zeros(C.size()).type_as(C.data)
            out_res = torch.zeros(lenText, self.nclass).type_as(feature.data)
            out_attns = torch.zeros(lenText, nH, nW).type_as(A.data)

            hidden = torch.zeros(nB, self.nchannel).type_as(C.data)
            prev_emb = self.char_embeddings.index_select(0, torch.zeros(nB).long().type_as(text.data))
            for i in range(0, nsteps):
                hidden = self.rnn(torch.cat((C[i, :, :], prev_emb), dim = 1),
                                 hidden)
                gru_res[i, :, :] = hidden
                prev_emb = self.char_embeddings.index_select(0, text[:, i])
            gru_res = self.generator(gru_res)

            start = 0
            for i in range(0, nB):
                cur_length = int(text_length[i])
                out_res[start : start + cur_length] = gru_res[0: cur_length,i,:]
                out_attns[start : start + cur_length] = A[i,0:cur_length,:,:]
                start += cur_length

            return out_res, out_attns

        else:
            lenText = nT
            nsteps = nT
            out_res = torch.zeros(lenText, nB, self.nclass).type_as(feature.data)

            hidden = torch.zeros(nB, self.nchannel).type_as(C.data)
            prev_emb = self.char_embeddings.index_select(0, torch.zeros(nB).long().type_as(text.data))
            out_length = torch.zeros(nB)
            now_step = 0
            while 0 in out_length and now_step < nsteps:
                hidden = self.rnn(torch.cat((C[now_step, :, :], prev_emb), dim = 1),
                                 hidden)
                tmp_result = self.generator(hidden)
                out_res[now_step] = tmp_result
                tmp_result = tmp_result.topk(1)[1].squeeze()
                for j in range(nB):
                    if out_length[j] == 0 and tmp_result[j] == 0:
                        out_length[j] = now_step + 1
                prev_emb = self.char_embeddings.index_select(0, tmp_result)
                now_step += 1
            for j in range(0, nB):
                if int(out_length[j]) == 0:
                    out_length[j] = nsteps

            start = 0
            output = torch.zeros(int(out_length.sum()), self.nclass).type_as(feature.data)
            for i in range(0, nB):
                cur_length = int(out_length[i])
                output[start : start + cur_length] = out_res[0: cur_length,i,:]
                start += cur_length

            return output, out_length
