import torch.nn as nn
from models.nlp.embedding.embedding import EmbeddingLayer, ConvolutionEmbeddingLayer
from models.nlp.transformer.encoder import EncoderLayer, Encoder
from models.nlp.transformer.decoder import DecoderLayer, Decoder
from utils import nn_utils


# only encoder
class HierarchicalTransformer(nn.Module):
    def __init__(self, n_tokens, max_seq_len, N=6, input_dim=512, hidden_dim=1024, h=8):
        super(HierarchicalTransformer, self).__init__()
        self.embedding = EmbeddingLayer(vocab_size=n_tokens, max_seq_len=max_seq_len, dim=input_dim)
        self.encoder = Encoder(EncoderLayer(h, input_dim, hidden_dim), N)

    def forward(self, X, mask):

        X = self.embedding(X)
        X = self.encoder(X, mask) # output
        return X