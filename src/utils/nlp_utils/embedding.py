import torch.nn as nn
from utils.nlp_utils.positional_encoding import PositionalEncoding


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, max_seq_len, dim):
        super(EmbeddingLayer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.positional_embedding = PositionalEncoding(dim, max_seq_len)

    def forward(self, X):
        tokens = self.token_embedding(X)
        return self.positional_embedding(tokens)