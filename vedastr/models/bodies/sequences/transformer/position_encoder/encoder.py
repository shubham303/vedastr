import torch.nn as nn

from .registry import POSITION_ENCODERS
from .utils import generate_encoder


@POSITION_ENCODERS.register_module
class PositionEncoder1D(nn.Module):

    def __init__(self, in_channels, max_len=2000, dropout=0.1):
        super(PositionEncoder1D, self).__init__()

        position_encoder = generate_encoder(in_channels, max_len)
        position_encoder = position_encoder.unsqueeze(0)
        self.register_buffer('position_encoder', position_encoder)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        
        if len(x.size()) ==4:
            # dimensions of feature extraction output and transformer decoder don't match.
            #size of tensor from Feat. extraction is 4 and from transformer decoder is 3.
            x=x.squeeze(2)
            # input order (B,E,C,T) -> (B, C,T, E)
            x = x.permute(0,2,1)
        out = x + self.position_encoder[:, :x.size(1), :]
        out = self.dropout(out)

        return out
