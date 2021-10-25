from vedastr.utils import build_from_cfg
from .registry import SEQUENCE_DECODERS, SEQUENCE_ENCODERS


def build_sequence_encoder(cfg, default_args=None):
    sequence_encoder = build_from_cfg(cfg, SEQUENCE_ENCODERS, default_args)

    return sequence_encoder


def build_sequence_decoder(cfg, default_args=None):
    sequence_encoder = build_from_cfg(cfg, SEQUENCE_DECODERS, default_args)

    return sequence_encoder

"""
below function is duplicate of build_sequence_encoder , it is defined to avoid circular dependency in DPT.
"""
def build_sequence_encoder1(cfg, default_args=None):
    sequence_encoder = build_from_cfg(cfg, SEQUENCE_ENCODERS, default_args)

    return sequence_encoder

