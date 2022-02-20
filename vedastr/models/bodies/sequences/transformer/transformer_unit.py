import logging

from torch import nn

from vedastr.models.bodies.sequences.registry import SEQUENCE_ENCODERS
from vedastr.models.bodies.sequences.transformer.embedding.builder import build_embedding_layer
from vedastr.models.bodies.sequences.transformer.position_encoder import build_position_encoder
from vedastr.models.bodies.sequences.transformer.unit import build_encoder_layer

logger = logging.getLogger()


@SEQUENCE_ENCODERS.register_module
class TransformerUnit(nn.Module):
	def __init__(
			self,
			encoder_layer,
			num_layers,
			position_encoder=None,
			embedding=None):
		
		super(TransformerUnit, self).__init__()
		
		if position_encoder is not None:
			self.pos_encoder = build_position_encoder(position_encoder)
		
		if embedding is not None:
			self.embedding = build_embedding_layer(embedding)
		
		self.layers = nn.ModuleList(
			[build_encoder_layer(encoder_layer) for _ in range(num_layers)])
	
	@property
	def with_position_encoder(self):
		return hasattr(self, 'pos_encoder') and self.pos_encoder is not None
	
	@property
	def with_embedding_layer(self):
		return hasattr(self, 'embedding') and self.embedding is not None
	
	def forward(self, query, key, value, src_mask=None):
		
		if self.with_embedding_layer:
			src, _ = self.embedding(query)
		
		if self.with_position_encoder:
			src = self.pos_encoder(query)
		
		for enc_layer in self.layers:
			enc_output = enc_layer(query, key, value, src_mask=src_mask)
		
		return enc_output
