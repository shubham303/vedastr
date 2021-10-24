import logging

import torch
from torch import nn
from ..registry import SEQUENCE_ENCODERS
from vedastr.models.utils import build_torch_nn

logger = logging.getLogger()

@SEQUENCE_ENCODERS.register_module
class DPT(nn.Module):
	def __init__(self, layers: list, last_embedding_dim, norm):
		super(DPT, self).__init__()
		self.encoders =[]# nn.ModuleList( build_sequence_encoder(encoder) for encoder in encoders)
		self.cls_token = nn.Parameter(torch.zeros(1, 1, last_embedding_dim))
		self.norm = build_torch_nn(norm)
		
	def forward(self,x):
		"""
		x : input
		src_masks : list of src_mask used at every layer.
		"""
		B = x.shape[0]
		
		for encoder in self.encoders[-1]:
			x = encoder(x)
		
		cls_tokens = self.cls_token.expand(B, -1, -1)
		x = torch.cat((cls_tokens, x), dim=1)
		
		x= self.encoders[-1]
		
		x =self.norm(x)
		
		return x[:, 0]