"""ref: https://github.com/Wang-Tianwei/Decoupled-attention-network"""
import logging
from typing import List

import torch
import torch.nn as nn

from vedastr.models.bodies.feature_extractors.encoders.backbones import resnet
from vedastr.models.bodies.feature_extractors.encoders.backbones.registry import BACKBONES

logger = logging.getLogger()




class CAM(nn.Module):
	def __init__(self, scales, maxT, depth, num_channels):
		super(CAM, self).__init__()
		# cascade multiscale features
		fpn = []
		for i in range(1, len(scales)):
			assert not (scales[i - 1][1] / scales[i][1]) % 1, 'layers scale error, from {} to {}'.format(i - 1, i)
			assert not (scales[i - 1][2] / scales[i][2]) % 1, 'layers scale error, from {} to {}'.format(i - 1, i)
			ksize = [3, 3, 5]  # if downsampling ratio >= 3, the kernel size is 5, else 3
			r_h, r_w = int(scales[i - 1][1] / scales[i][1]), int(scales[i - 1][2] / scales[i][2])
			ksize_h = 1 if scales[i - 1][1] == 1 else ksize[r_h - 1]
			ksize_w = 1 if scales[i - 1][2] == 1 else ksize[r_w - 1]
			fpn.append(nn.Sequential(nn.Conv2d(scales[i - 1][0], scales[i][0],
			                                   (ksize_h, ksize_w),
			                                   (r_h, r_w),
			                                   (int((ksize_h - 1) / 2), int((ksize_w - 1) / 2))),
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
			stride = [2] if 2 ** (depth / 2 - i) <= h else [1]
			stride = stride + [2] if 2 ** (depth / 2 - i) <= w else stride + [1]
			strides.append(stride)
			conv_ksizes.append([3, 3])
			deconv_ksizes.append([_ ** 2 for _ in stride])
		convs = [nn.Sequential(nn.Conv2d(in_shape[0], num_channels,
		                                 tuple(conv_ksizes[0]),
		                                 tuple(strides[0]),
		                                 (int((conv_ksizes[0][0] - 1) / 2), int((conv_ksizes[0][1] - 1) / 2))),
		                       nn.BatchNorm2d(num_channels),
		                       nn.ReLU(True))]
		for i in range(1, int(depth / 2)):
			convs.append(nn.Sequential(nn.Conv2d(num_channels, num_channels,
			                                     tuple(conv_ksizes[i]),
			                                     tuple(strides[i]),
			                                     (int((conv_ksizes[i][0] - 1) / 2), int((conv_ksizes[i][1] - 1) / 2))),
			                           nn.BatchNorm2d(num_channels),
			                           nn.ReLU(True)))
		self.convs = nn.Sequential(*convs)
		# deconvs
		deconvs = []
		
		for i in range(1, int(depth / 2)):
			deconvs.append(nn.Sequential(nn.ConvTranspose2d(num_channels, num_channels,
			                                                tuple(deconv_ksizes[int(depth / 2) - i]),
			                                                tuple(strides[int(depth / 2) - i]),
			                                                (int(deconv_ksizes[int(depth / 2) - i][0] / 4.),
			                                                 int(deconv_ksizes[int(depth / 2) - i][1] / 4.))),
			                             nn.BatchNorm2d(num_channels),
			                             nn.ReLU(True)))
		deconvs.append(nn.Sequential(nn.ConvTranspose2d(num_channels, maxT,
		                                                tuple(deconv_ksizes[0]),
		                                                tuple(strides[0]),
		                                                (int(deconv_ksizes[0][0] / 4.), int(deconv_ksizes[0][1] / 4.))),
		                             nn.Sigmoid()))
		self.deconvs = nn.Sequential(*deconvs)
	
	def forward(self, input: List[torch.Tensor]):
		x = input[0]
		for i , layer in enumerate(self.fpn):
			x = layer(x) + input[i + 1]
		conv_feats = []
		for i , layer in enumerate(self.convs):
			x = layer(x)
			conv_feats.append(x)
		for i , layer in  enumerate(self.deconvs[:-1]):
			x = layer(x)
			x = x + conv_feats[len(conv_feats) - 2 - i]
		x = self.deconvs[-1](x)
		return x

@BACKBONES.register_module
class FPN(nn.Module):
	def __init__(self,strides, compress_layer, input_shape, maxT, depth, num_channels ):
		super(FPN, self).__init__()
		self.model = resnet.resnet45(strides, compress_layer)
		self.input_shape = input_shape
		self.CAM = CAM(self.Iwantshapes(), maxT, depth,num_channels)
	
	def forward(self, x):
		feats={}
		x = self.model(x)
		A = self.CAM(x)
		nB, nC, nH, nW = x[-1].size()
		nT = A.size()[1]
		# Normalize
		A = A / A.view(nB, nT, -1).sum(2).view(nB, nT, 1, 1)
		
		#print attention maps
		"""q = A.cpu().detach().numpy()[0]
		from matplotlib import pyplot as plt
		
		for i in range(len(q)) :
			plt.imshow(q[i])
			plt.savefig("attn_{}.png".format(i))"""
		
		# weighted sum
		C = x[-1].view(nB, 1, nC, nH, nW) * A.view(nB, nT, 1, nH, nW)
		C = C.view(nB, nT, nC, -1).sum(3)
		feats["c1"] = C
		return feats
	
	def Iwantshapes(self):
		pseudo_input = torch.rand(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
		features = self.model(pseudo_input)
		return [feat.size()[1:] for feat in features]


	
