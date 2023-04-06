import torch.nn as nn
from prediction.models.nlp.embedding import EmbeddingLayer
from prediction.models.nlp.encoder import EncoderLayer, Encoder


class ClassificationTransformer(nn.Module):
    def __init__(self, n_tokens, seq_len, n_classes, N=6, d=512, d_ff=2048, h=8):
        super(ClassificationTransformer, self).__init__()
        self.embedding = EmbeddingLayer(vocab_size=n_tokens, seq_len=seq_len, dim=d)
        self.encoder = Encoder(EncoderLayer(h, d, d_ff), N)
        self.linear = nn.Linear(d, n_classes)

    def forward(self, X):
        X = self.embedding(X)
        X = self.encoder(X)
        # pool the embeddings of all tokens using mean
        X = X.mean(dim=1)

        y = self.linear(X)
        return y