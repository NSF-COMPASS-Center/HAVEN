from utils import nn_utils
from models.nlp.transformer.multi_head_attention import MultiHeadAttention
from models.nlp.transformer.feed_forward_layer import FeedForwardLayer
from models.nlp.transformer.layer_normalization import LayerNormalization
from models.nlp.transformer.residual_connection import ResidualConnectionLayer
import torch.nn as nn


class DecoderLayer(nn.Module):
    def __init__(self, h=8, d=512, d_ff=2048):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(h, d)
        self.cross_attn = MultiHeadAttention(h, d)
        self.feed_forward = FeedForwardLayer(d, d_ff)
        self.residual_connections = nn_utils.create_clones(ResidualConnectionLayer(), 3)

    # source_emb is the embedding from the encoder layer
    def forward(self, X, source_emb, source_mask, target_mask):
        # decoder multi-head attention
        # mask is the target mask
        X = self.residual_connections[0](X, lambda X: self.self_attn(X, X, X, target_mask))

        # cross-attention with key and values from source_emb and query from the previous layer in the decoder
        # mask is the source_mask
        X = self.residual_connections[1](X, lambda X: self.cross_attn(X, source_emb, source_emb, source_mask))

        return self.residual_connections[2](X, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N=6):
        super(Decoder, self).__init__()
        self.layers = nn_utils.create_clones(layer, N)
        self.layer_norm = LayerNormalization()
        self.encoding = None

    def forward(self, X, source_emb, source_mask, target_mask):
        # pass through each layer sequentially
        for layer in self.layers:
            X = layer(X, source_emb, source_mask, target_mask)
        self.decoding = self.layer_norm(X)
        return self.decoding