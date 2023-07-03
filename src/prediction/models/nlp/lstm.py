import torch
import torch.nn as nn
from utils.nlp_utils.embedding import EmbeddingLayer
from torch.nn import LSTM
from utils import nn_utils


class LSTM_Model(nn.Module):
    def __init__(self, n_tokens, max_seq_len, n_classes, N, input_dim, hidden_dim):
        super(LSTM_Model, self).__init__()
        # assuming hidden state dimension = cell state dimension = hidden_dim
        self.hidden_dim = hidden_dim
        self.N = N
        self.embedding = EmbeddingLayer(vocab_size=n_tokens, max_seq_len=max_seq_len, dim=input_dim)
        # assuming hidden state dimension = cell state dimension = output_dimension = hidden_dim and projection_size=0
        self.lstm = LSTM(input_size=input_dim,
                         hidden_size=hidden_dim,
                         num_layers=N,
                         batch_first=True)
        self.linear = nn.Linear(hidden_dim, n_classes)

    def forward(self, X):
        X = self.embedding(X)
        hidden_input = self.init_zeros(batch_size=X.size(0))
        cell_input = self.init_zeros(batch_size=X.size(0))

        # return values from rnn:
        # output: output features from the last layer for each token: num_lstm_layers x batch_size X sequence_length X hidden_dim
        # hidden_output: final hidden state (embedding) for each sequence: num_lstm_layers x batch_size X hidden_dim
        # cell_output: final cell state (embedding) for each sequence: num_lstm_layers x batch_size X hidden_dim
        output, (hidden_output, cell_output) = self.lstm(X, (hidden_input, cell_input))

        # aggregate the embeddings from lstm
        # mean of the representations of all tokens
        lstm_emb = output.squeeze().mean(dim=1)
        y = self.linear(lstm_emb)
        return y

    def init_zeros(self, batch_size):
        # dimensions: N (num of lstm layers) X batch_size X hidden_layer_dimension
        return torch.zeros(self.N, batch_size, self.hidden_dim).to(nn_utils.get_device())


def get_lstm_model(model):
    lstm_model = LSTM_Model(n_tokens=model["n_tokens"],
                            max_seq_len=model["max_seq_len"],
                            n_classes=model["n_classes"],
                            N=model["depth"],
                            input_dim=model["input_dim"],
                            hidden_dim=model["hidden_dim"])

    print(lstm_model)
    print("Number of parameters = ", sum(p.numel() for p in lstm_model.parameters() if p.requires_grad))
    return lstm_model.to(nn_utils.get_device())
