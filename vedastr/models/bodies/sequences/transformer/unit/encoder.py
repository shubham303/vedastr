import torch.nn as nn

from vedastr.models.utils import build_torch_nn
from .attention import build_attention
from .feedforward import build_feedforward
from .registry import TRANSFORMER_ENCODER_LAYERS


class _TransformerEncoderLayer(nn.Module):

    def __init__(self, attention, attention_norm, feedforward,
                 feedforward_norm):
        super(_TransformerEncoderLayer, self).__init__()
        self.attention = build_attention(attention)
        self.attention_norm = build_torch_nn(attention_norm)

        self.feedforward = build_feedforward(feedforward)
        self.feedforward_norm = build_torch_nn(feedforward_norm)


@TRANSFORMER_ENCODER_LAYERS.register_module
class TransformerEncoderLayer1D(_TransformerEncoderLayer):

    def __init__(self, attention, attention_norm, feedforward,
                 feedforward_norm):
        super(TransformerEncoderLayer1D,
              self).__init__(attention, attention_norm, feedforward,
                             feedforward_norm)

    def forward(self,query,key, value, src_mask=None):
        attn_out, _ = self.attention( query,key, value, src_mask)
        out1 = self.attention_norm(query + attn_out)

        ffn_out = self.feedforward(out1)
        out2 = self.feedforward_norm(out1 + ffn_out)

        return out2


@TRANSFORMER_ENCODER_LAYERS.register_module
class TransformerEncoderLayer2D(_TransformerEncoderLayer):

    def __init__(self, attention, attention_norm, feedforward,
                 feedforward_norm):
        super(TransformerEncoderLayer2D,
              self).__init__(attention, attention_norm, feedforward,
                             feedforward_norm)

    def norm(self, norm_layer, x):
        b, c, h, w = x.size()

        if isinstance(norm_layer, nn.LayerNorm):
            out = x.view(b, c, h * w).transpose(1, 2)
            out = norm_layer(out)
            out = out.transpose(1, 2).contiguous().view(b, c, h, w)
        else:
            out = norm_layer(x)

        return out

    def forward(self, query,key, value, src_mask=None):
        b, c, h, w = query.size()

        query = query.view(b, c, h * w).transpose(1, 2)
        if src_mask is not None:
            src_mask = src_mask.view(b, 1, h * w)

        attn_out, _ = self.attention(query, key, value, src_mask)
        out1 = query + attn_out
        out1 = out1.transpose(1, 2).contiguous().view(b, c, h, w)
        out1 = self.norm(self.attention_norm, out1)

        ffn_out = self.feedforward(out1)
        out2 = self.norm(self.feedforward_norm, out1 + ffn_out)

        return out2
