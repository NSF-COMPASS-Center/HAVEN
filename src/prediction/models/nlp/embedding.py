import torch
import torch.nn as nn
from utils import nn_utils


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, seq_len, dim):
        super(EmbeddingLayer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.positional_embedding = nn.Embedding(seq_len, dim)

    def forward(self, X):
        tokens = self.token_embedding(X)
        b, n, dim = tokens.size()
        positions = self.positional_embedding(torch.arange(n, device=nn_utils.get_device()))[None, :, :].expand(b, n, dim)
        return tokens + positions