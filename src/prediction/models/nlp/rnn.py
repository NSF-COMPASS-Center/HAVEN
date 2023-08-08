import torch
import torch.nn as nn
from utils.nlp_utils.embedding import EmbeddingLayer
from torch.nn import RNN
from utils import nn_utils


class RNN_Model(nn.Module):
    def __init__(self, n_tokens, max_seq_len, n_classes, N, input_dim, hidden_dim):
        super(RNN_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.N = N
        self.embedding = EmbeddingLayer(vocab_size=n_tokens, max_seq_len=max_seq_len, dim=input_dim)
        self.rnn = RNN(input_size=input_dim,
                       hidden_size=hidden_dim,
                       num_layers=N,
                       nonlinearity="tanh",
                       batch_first=True)
        self.linear = nn.Linear(hidden_dim, n_classes)

    def forward(self, X):
        X = self.embedding(X)
        hidden_input = self.init_hidden(batch_size=X.size(0))

        # return values from rnn:
        # output: output features from the last layer for each token: num_rnn_layers x batch_size X sequence_length X hidden_dim
        # hidden_output: final hidden state (embedding) for each sequence: num_rnn_layers x batch_size X hidden_dim
        # the hidden output is essentially the hidden state of the last token of the sequence
        output, hidden_output = self.rnn(X, hidden_input)

        # aggregate the embeddings from rnn
        # mean of the representations of all tokens
        self.rnn_emb = output.squeeze().mean(dim=1)
        y = self.linear(self.rnn_emb)
        return y

    def init_hidden(self, batch_size):
        # dimensions: N (num of rnn layers) X batch_size X hidden_layer_dimension
        return torch.zeros(self.N, batch_size, self.hidden_dim).to(nn_utils.get_device())


def get_rnn_model(model):
    rnn_model = RNN_Model(n_tokens=model["n_tokens"],
                          max_seq_len=model["max_seq_len"],
                          n_classes=model["n_classes"],
                          N=model["depth"],
                          input_dim=model["input_dim"],
                          hidden_dim=model["hidden_dim"])

    print(rnn_model)
    print("Number of parameters = ", sum(p.numel() for p in rnn_model.parameters() if p.requires_grad))
    return rnn_model.to(nn_utils.get_device())
