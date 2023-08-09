import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.nlp_utils.embedding import EmbeddingLayer
from torch.nn import Conv1d
from utils import nn_utils


class CNN_Model(nn.Module):
    def __init__(self, n_tokens, max_seq_len, n_classes, N, input_dim, hidden_dim, kernel_size, stride):
        super(CNN_Model, self).__init__()
        self.embedding = EmbeddingLayer(vocab_size=n_tokens, max_seq_len=max_seq_len, dim=input_dim)
        self.conv1d = Conv1d(in_channels=input_dim,
                             out_channels=hidden_dim,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=0)
        self.conv1d_hidden = Conv1d(in_channels=hidden_dim,
                             out_channels=hidden_dim,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=0)
        # intermediate hidden layers (number = N-1): hidden_dim --> hidden_dim
        # N-1 because we already have one layer converting input_dim --> hidden_dim
        self.conv1d_hidden_layers = nn_utils.create_clones(self.conv1d_hidden, N-1)
        self.linear = nn.Linear(hidden_dim, n_classes)

    def forward(self, X):
        X = self.embedding(X)  # b x n x d
        X = torch.einsum("bnd->bdn", X)  # b x d x n (conv1d requires number of channels as the second dimension)
        X = self.conv1d(X)
        X = torch.einsum("bdn -> bnd", X)  # revert back to b x n x d (batch x updated length of sequence x output_channels dimension)
        X = F.relu(X)  # activation function

        for conv1d_hidden_layer in self.conv1d_hidden_layers:
            X = torch.einsum("bnd->bdn", X)  # b x d x n (conv1d requires number of channels as the second dimension)
            X = conv1d_hidden_layer(X)
            X = torch.einsum("bdn -> bnd", X)  # revert back to b x n x d (batch x updated length of sequence x output_channels dimension)
            X = F.relu(X)  # activation function

        # aggregate the embeddings from cnn
        # mean of the representations of all tokens
        self.cnn_emb = X.squeeze().mean(dim=1)
        y = self.linear(self.cnn_emb)
        return y


def get_cnn_model(model):
    cnn_model = CNN_Model(n_tokens=model["n_tokens"],
                          max_seq_len=model["max_seq_len"],
                          n_classes=model["n_classes"],
                          N = model["depth"],
                          input_dim=model["input_dim"],
                          hidden_dim=model["hidden_dim"],
                          kernel_size=model["kernel_size"],
                          stride=model["stride"])

    print(cnn_model)
    print("Number of parameters = ", sum(p.numel() for p in cnn_model.parameters() if p.requires_grad))
    return cnn_model.to(nn_utils.get_device())
