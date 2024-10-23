from utils import nn_utils
from models.baseline.nlp.transformer import MultiHeadAttention
from models.baseline.nlp.transformer import FeedForwardLayer
from models.baseline.nlp.transformer import LayerNormalization
from models.baseline.nlp.transformer import ResidualConnectionLayer
import torch.nn as nn


class EncoderLayer(nn.Module):
    def __init__(self, h=8, d=512, d_ff=2048):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(h, d)
        self.feed_forward = FeedForwardLayer(d, d_ff)
        self.residual_connections = nn_utils.create_clones(ResidualConnectionLayer(), 2)

    def forward(self, X, mask):
        X = self.residual_connections[0](X, lambda X: self.self_attn(X, X, X, mask))
        return self.residual_connections[1](X, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer, N=6):
        super(Encoder, self).__init__()
        self.layers = nn_utils.create_clones(layer, N)
        self.layer_norm = LayerNormalization()
        self.encoding = None

    def forward(self, X, mask=None):
        # pass through each layer sequentially
        for layer in self.layers:
            X = layer(X, mask)
        self.encoding = self.layer_norm(X)
        return self.encoding
