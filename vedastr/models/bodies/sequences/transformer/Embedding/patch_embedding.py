import torch
from einops import einops
from einops.layers.torch import Rearrange
from torch import nn

from vedastr.models.bodies.sequences.transformer.Embedding.registry import EMBEDDING

"""
Ref : https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632
"""

@EMBEDDING.register_module
class PatchEmbedding(nn.Module):
	"""
	this is convolution based patch embedding.
	"""
	def __init__(self, in_channels: int = 3, patch_height=4, patch_width = 4, emb_size=512):
		self.patch_height=patch_height
		self.patch_width=patch_width
		super().__init__()
		self.projection = nn.Sequential(
			# using a conv layer instead of a linear one -> performance gains
			nn.Conv2d(in_channels, emb_size, kernel_size=(patch_height,patch_width), stride=patch_width),
			Rearrange('b e (h) (w) -> b (h w) e'),
		)
		self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
	
	def forward(self, x):
		b, _, _, _ = x.shape
		x = self.projection(x)
		cls_token = einops.repeat(self.cls_token, '() n e -> b n e' , b=b )
		x = torch.cat([cls_token , x],dim=1)
		return x
