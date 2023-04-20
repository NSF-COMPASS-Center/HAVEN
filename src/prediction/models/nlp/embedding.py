import torch.nn as nn
from prediction.models.nlp.positional_encoding import PositionalEncoding


class EmbeddingLayer(nn.Module):
    # def __init__(self, vocab_size, max_seq_len, dim):
    #     super(EmbeddingLayer, self).__init__()
    #     self.token_embedding = nn.Embedding(vocab_size, dim)
    #     self.positional_embedding = nn.Embedding(seq_len, dim)
    #
    # def forward(self, X):
    #     tokens = self.token_embedding(X)
    #     b, n, dim = tokens.size()
    #     positions = self.positional_embedding(torch.arange(n, device=nn_utils.get_device()))[None, :, :].expand(b, n, dim)
    #     return tokens + positions

    def __init__(self, vocab_size, max_seq_len, dim):
        super(EmbeddingLayer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.positional_embedding = PositionalEncoding(dim, max_seq_len)

    def forward(self, X):
        tokens = self.token_embedding(X)
        return self.positional_embedding(tokens)
