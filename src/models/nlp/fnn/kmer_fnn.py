import torch.nn as nn
import torch.nn.functional as F
from utils import nn_utils


class Kmer_FNN_Model(nn.Module):
    def __init__(self, n_classes, N, input_dim, hidden_dim):
        super(Kmer_FNN_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.N = N

        # first linear layer: input_dim --> hidden_dim
        self.linear_ip = nn.Linear(input_dim, hidden_dim)

        self.linear_hidden = nn.Linear(hidden_dim, hidden_dim)
        # intermediate hidden layers (number = N): hidden_dim --> hidden_dim
        self.linear_hidden_n = nn_utils.create_clones(self.linear_hidden, N)

        # last linear layer: hidden_dim--> n_classes
        self.linear_op = nn.Linear(hidden_dim, n_classes)

    def forward(self, X):
        # input linear layer
        X = F.relu(self.linear_ip(X))
        # hidden
        for linear_layer in self.linear_hidden_n:
            X = F.relu(linear_layer(X))
        y = self.linear_op(X)
        return y


def get_fnn_model(model):
    fnn_model = Kmer_FNN_Model(n_classes=model["n_classes"],
                          N=model["depth"],
                          input_dim=model["input_dim"],
                          hidden_dim=model["hidden_dim"])

    print(fnn_model)
    print("Number of parameters = ", sum(p.numel() for p in fnn_model.parameters() if p.requires_grad))
    return fnn_model.to(nn_utils.get_device())
