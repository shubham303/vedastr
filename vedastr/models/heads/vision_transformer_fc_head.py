import logging

import torch.nn as nn

from vedastr.models.utils import build_torch_nn
from vedastr.models.weight_init import init_weights
from .registry import HEADS

logger = logging.getLogger()


@HEADS.register_module
class VisionTransformerFChead(nn.Module):
    """VisionTransformerFChead
    """
    def __init__(
            self,
            in_channels,
            num_class,
            from_layer,
            pool=None,
            export=False,
		    batch_max_length=25
    ):
        super(VisionTransformerFChead, self).__init__()

        self.num_class = num_class
        self.from_layer = from_layer
        fc = nn.Linear(in_channels, num_class)
        if pool is not None:
            self.pool = build_torch_nn(pool)
        self.fc = fc
        self.export = export
        self.batch_max_len =batch_max_length
        logger.info('CTCHead init weights')
        init_weights(self.modules())

    @property
    def with_pool(self):
        return hasattr(self, 'pool') and self.pool is not None

    def forward(self, x_input):
        x = x_input[self.from_layer]
        x=x[:, 1:self.batch_max_len+2]
        """if self.export:
            x = x.mean(2).permute(0, 2, 1)
        elif self.with_pool:
            x = self.pool(x).permute(0, 3, 1, 2).squeeze(3)"""
        b, s, e = x.size()
        x = x.reshape(b * s, e)
        x = self.fc(x).view(b, s, self.num_class)
        return x
