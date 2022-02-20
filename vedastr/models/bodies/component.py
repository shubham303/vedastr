import torch.nn as nn

from .feature_extractors import build_brick, build_feature_extractor
from .rectificators import build_rectificator
from .registry import COMPONENT
from .sequences import build_sequence_encoder
from .sequences.transformer.embedding.builder import build_embedding_layer
from .sequences.transformer.position_encoder import build_position_encoder
from ..utils import build_module


class BaseComponent(nn.Module):

    def __init__(self, from_layer, to_layer, component):
        super(BaseComponent, self).__init__()

        self.from_layer = from_layer
        self.to_layer = to_layer
        self.component = component

    def forward(self, x):
        return self.component(x)


@COMPONENT.register_module
class FeatureExtractorComponent(BaseComponent):

    def __init__(self, from_layer, to_layer, arch):
        super(FeatureExtractorComponent,
              self).__init__(from_layer, to_layer,
                             build_feature_extractor(arch))


@COMPONENT.register_module
class RectificatorComponent(BaseComponent):

    def __init__(self, from_layer, to_layer, arch):
        super(RectificatorComponent, self).__init__(from_layer, to_layer,
                                                    build_rectificator(arch))


@COMPONENT.register_module
class SequenceEncoderComponent(BaseComponent):

    def __init__(self, from_layer, to_layer, arch):
        super(SequenceEncoderComponent,
              self).__init__(from_layer, to_layer,
                             build_sequence_encoder(arch))


@COMPONENT.register_module
class BrickComponent(BaseComponent):

    def __init__(self, from_layer, to_layer, arch):
        super(BrickComponent, self).__init__(from_layer, to_layer,
                                             build_brick(arch))


@COMPONENT.register_module
class PlugComponent(BaseComponent):

    def __init__(self, from_layer, to_layer, arch):
        super(PlugComponent, self).__init__(from_layer, to_layer,
                                            build_module(arch))


@COMPONENT.register_module
class PositionalEncodingComponent(BaseComponent):
    def __init__(self, from_layer, to_layer, arch):
        super(PositionalEncodingComponent, self).__init__(from_layer, to_layer,
                                            build_position_encoder(arch))


@COMPONENT.register_module
class EmbeddingComponent(BaseComponent):
    def __init__(self, from_layer, to_layer, arch):
        super(EmbeddingComponent, self).__init__(from_layer, to_layer,
                                            build_embedding_layer(arch))