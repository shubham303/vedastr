from vedastr.utils import build_from_cfg
from .registry import BOX_CODER


def build_box_coder(cfg, default_args=None):
    sequence_encoder = build_from_cfg(cfg, BOX_CODER, default_args)
    return sequence_encoder