import torch

from vedastr.models.utils import build_torch_nn
from vedastr.utils import build_from_cfg
from vedastr.models.bodies.sequences.transformer.embedding.registry import EMBEDDING

def build_embedding_layer(cfg, default_args=None):
	type = cfg["type"]
	
	if hasattr(torch.nn, type):
		return build_torch_nn(cfg, default_args)
	
	return  build_from_cfg(cfg, EMBEDDING , default_args)

