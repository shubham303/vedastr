from vedastr.utils import build_from_cfg
from .registry import CDISNET_BODY


def build_cdisnet_body(cfg, default_args=None):
    cdisnet_body = build_from_cfg(cfg, CDISNET_BODY, default_args)

    return cdisnet_body
