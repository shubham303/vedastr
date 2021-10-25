import logging

import torch
from torch import nn

from ..builder import build_sequence_encoder1
from ..registry import SEQUENCE_ENCODERS
from vedastr.models.utils import build_torch_nn

logger = logging.getLogger()

@SEQUENCE_ENCODERS.register_module
class DPT(nn.Module):
	def __init__(self, layers: list, norm):
		super(DPT, self).__init__()
		self.encoders = nn.ModuleList(build_sequence_encoder1(encoder) for encoder in layers)

		self.norm = build_torch_nn(norm)
		
	def forward(self,x):
		"""
		x : input
		src_masks : list of src_mask used at every layer.
		"""
		for i in range(0,4):
			x = self.encoders[i](x)
		
		x =self.norm(x)
		
		return x