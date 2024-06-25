import torch
import torch.nn as nn
import torch.nn.functional as F
from models.nlp.embedding.positional_encoding import PositionalEncoding


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, max_seq_len, dim):
        super(EmbeddingLayer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.positional_embedding = PositionalEncoding(dim, max_seq_len)

    def forward(self, X):
        tokens = self.token_embedding(X.long())
        return self.positional_embedding(tokens)


class ConvolutionEmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, max_seq_len, dim, kernel_size, stride, padding):
        super(ConvolutionEmbeddingLayer, self).__init__()
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.conv1d = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, stride=stride,
                                padding=padding)
        self.positional_embedding = PositionalEncoding(dim, max_seq_len)

    def forward(self, X):
        X = self.token_embedding(X.long()) # b x n x d
        X = self.positional_embedding(X)
        # convolution layer
        X = torch.einsum("bnd->bdn", X)
        X = self.conv1d(X)
        X = torch.einsum("bdn->bnd", X)
        return X

