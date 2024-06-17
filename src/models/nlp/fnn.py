import torch.nn as nn
import torch.nn.functional as F
from models.nlp.embedding.embedding import EmbeddingLayer
from utils import nn_utils


class FNN_Model(nn.Module):
    def __init__(self, n_tokens, max_seq_len, n_classes, N, input_dim, hidden_dim):
        super(FNN_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.N = N
        self.embedding = EmbeddingLayer(vocab_size=n_tokens, max_seq_len=max_seq_len, dim=input_dim)
        # first linear layer: input_dim --> hidden_dim
        self.linear_ip = nn.Linear(input_dim, hidden_dim)

        self.linear_hidden = nn.Linear(hidden_dim, hidden_dim)
        # intermediate hidden layers (number = N): hidden_dim --> hidden_dim
        self.linear_hidden_n = nn_utils.create_clones(self.linear_hidden, N)

        # last linear layer: hidden_dim--> n_classes
        self.linear_op = nn.Linear(hidden_dim, n_classes)

    def get_embedding(self, X):
        X = self.embedding(X)
        # input linear layer
        X = F.relu(self.linear_ip(X))
        # hidden
        for linear_layer in self.linear_hidden_n:
            X = F.relu(linear_layer(X))
        # mean of the representations of all tokens
        return X.mean(dim=1)

    def forward(self, X):
        self.input_embedding = self.get_embedding(X)
        y = self.linear_op(self.input_embedding)
        return y


def get_fnn_model(model):
    fnn_model = FNN_Model(n_tokens=model["n_tokens"],
                          max_seq_len=model["max_seq_len"],
                          n_classes=model["n_classes"],
                          N=model["depth"],
                          input_dim=model["input_dim"],
                          hidden_dim=model["hidden_dim"])

    print(fnn_model)
    print("Number of parameters = ", sum(p.numel() for p in fnn_model.parameters() if p.requires_grad))
    return fnn_model.to(nn_utils.get_device())
