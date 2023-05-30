import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import nn_utils
from prediction.models.nlp.positional_encoding import PositionalEncoding


# class EmbeddingLayer(nn.Module):
#     def __init__(self, vocab_size, max_seq_len, dim):
#         super(EmbeddingLayer, self).__init__()
#         self.token_embedding = nn.Embedding(vocab_size, dim)
#         self.positional_embedding = nn.Embedding(max_seq_len, dim)
#
#     def forward(self, X):
#         tokens = self.token_embedding(X)
#         b, n, dim = tokens.size()
#         positions = self.positional_embedding(torch.arange(n, device=nn_utils.get_device()))[None, :, :].expand(b, n, dim)
#         return tokens + positions


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, max_seq_len, dim):
        super(EmbeddingLayer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.positional_embedding = PositionalEncoding(dim, max_seq_len)

    def forward(self, X):
        tokens = self.token_embedding(X)
        return self.positional_embedding(tokens)


# class ConvolutionEmbeddingLayer(nn.Module):
#     def __init__(self, vocab_size, max_seq_len, dim, kernel_size, stride, padding):
#         super(ConvolutionEmbeddingLayer, self).__init__()
#         self.vocab_size = vocab_size
#         self.linear = nn.Linear(vocab_size, dim)
#         self.conv1d = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, stride=stride,
#                                 padding=padding)
#         self.positional_embedding = PositionalEncoding(dim, max_seq_len)
#
#     def forward(self, X):
#         X = F.one_hot(X.to(torch.int64), self.vocab_size).float()  # b x n x vocab_size where n is the number of tokens
#         X = self.linear(X) # b x n x d
#         X = torch.einsum("bnd->bdn", X)
#         X = self.conv1d(X)
#         X = torch.einsum("bdn->bnd", X)
#         return self.positional_embedding(X)


# Convolution with Linear Embedding
class ConvolutionEmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, max_seq_len, dim, kernel_size, stride, padding):
        super(ConvolutionEmbeddingLayer, self).__init__()
        self.vocab_size = vocab_size
        self.linear = nn.Linear(vocab_size, dim)
        self.conv1d = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, stride=stride,
                                padding=padding)
        self.positional_embedding = nn.Embedding(max_seq_len, dim)

    def forward(self, X):
        X = F.one_hot(X.to(torch.int64), self.vocab_size).float() # b x n x vocab_size where n is the number of tokens
        X = self.linear(X)  # b x n x d
        X = torch.einsum("bnd->bdn", X)
        X = self.conv1d(X)
        tokens = torch.einsum("bdn->bnd", X)
        b, n, dim = tokens.size()
        positions = self.positional_embedding(torch.arange(n, device=nn_utils.get_device()))[None, :, :].expand(b, n,
                                                                                                                dim)
        return tokens + positions
