import logging

import torch
import torch.nn as nn

from vedastr.models.bodies.sequences.transformer.position_encoder import build_position_encoder
from vedastr.models.weight_init import init_weights
from .embedding.builder import build_embedding_layer
from .unit import build_encoder_layer
from ..registry import SEQUENCE_ENCODERS

logger = logging.getLogger()


@SEQUENCE_ENCODERS.register_module
class TransformerEncoder(nn.Module):
	
	def __init__(self,
	             encoder_layer: dict,
	             num_layers: int,
	             position_encoder: dict = None,
	             embedding: dict = None,
	             has_cls_token = False):
		super(TransformerEncoder, self).__init__()
		
		if position_encoder is not None:
			self.pos_encoder = build_position_encoder(position_encoder)
		
		if embedding is not None:
			self.embedding = build_embedding_layer(embedding)
		
		self.layers = nn.ModuleList(
			[build_encoder_layer(encoder_layer) for _ in range(num_layers)])
		
		self.has_cls_token=has_cls_token
		if self.has_cls_token:
			self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embedding.embed_dim))
		logger.info('TransformerEncoder init weights')
		init_weights(self.modules())
	
	@property
	def with_position_encoder(self):
		return hasattr(self, 'pos_encoder') and self.pos_encoder is not None
	
	@property
	def with_embedding_layer(self):
		return hasattr(self, 'embedding') and self.embedding is not None
	
	def forward(self, src, src_mask=None):
		if self.with_embedding_layer:
			src,_ = self.embedding(src)
		
		if self.has_cls_token:
			B = src.shape[0]
			cls_tokens = self.cls_token.expand(B, -1, -1)
			src = torch.cat((cls_tokens, src), dim=1)
			
		if self.with_position_encoder:
			src = self.pos_encoder(src)
		
		for layer in self.layers:
			src = layer(src, src_mask)
		
		return src
