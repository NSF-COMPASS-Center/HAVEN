import torch.nn as nn
from utils.nlp_utils.embedding import EmbeddingLayer
from utils import nn_utils


class Transformer(nn.Module):
    def __init__(self, n_tokens, max_seq_len, n_classes, N=6, d=512, d_ff=1024, h=8):
        super(Transformer, self).__init__()
        # self.embedding = EmbeddingLayer(vocab_size=n_tokens, max_seq_len=max_seq_len, dim=d)
        # self.encoder = Encoder(EncoderLayer(h, d, d_ff), N)
        # self.linear = nn.Linear(d, n_classes)

    def forward(self, X):
        # X = self.embedding(X)
        # X = self.encoder(X)
        # # pool the embeddings of all tokens using mean
        # X = X.mean(dim=1)
        #
        # y = self.linear(X)
        # return y
        return None


def get_transformer_model(model):
    tf_model = Transformer(n_tokens=model["n_tokens"],
                           max_seq_len=model["max_seq_len"],
                           n_classes=model["n_classes"],
                           N=model["depth"],
                           d=model["dim"],
                           d_ff=2048,
                           h=model["n_heads"])

    print(tf_model)
    print("Number of parameters = ", sum(p.numel() for p in tf_model.parameters() if p.requires_grad))
    # print(f"Weight initialization = {model['weight_initialization']}")
    return tf_model.to(nn_utils.get_device())
