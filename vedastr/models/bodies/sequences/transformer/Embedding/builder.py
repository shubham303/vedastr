from vedastr.utils import build_from_cfg
from .patch_embedding import PatchEmbedding
from .registry import EMBEDDING

def build_embedding_layer(cfg, default_args=None):
	patch_embedding = build_from_cfg(cfg, EMBEDDING , default_args)
	return patch_embedding

