from utils import nn_utils
from prediction.models.nlp.multi_head_attention import MultiHeadAttention
from prediction.models.nlp.feed_forward_layer import FeedForwardLayer
from prediction.models.nlp.normalization import NormalizationLayer
from prediction.models.nlp.residual_connection import ResidualConnectionLayer
import torch.nn as nn


class EncoderLayer(nn.Module):
    def __init__(self, h=8, d=512, d_ff=2048):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(h, d)
        self.feed_forward = FeedForwardLayer(d, d_ff)
        self.residual_connections = nn_utils.create_clones(ResidualConnectionLayer(), 2)

    def forward(self, X):
        X = self.residual_connections[0](X, lambda X: self.self_attn(X, X, X))
        return self.residual_connections[1](X, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer, N=6):
        super(Encoder, self).__init__()
        self.layers = nn_utils.create_clones(layer, N)
        self.norm = NormalizationLayer()

    def forward(self, X):
        # pass through each layer sequentially
        for layer in self.layers:
            X = layer(X)
        return self.norm(X)
