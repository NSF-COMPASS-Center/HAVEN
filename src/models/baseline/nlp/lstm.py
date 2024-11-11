import torch
import torch.nn as nn
from models.baseline.nlp.embedding.embedding import EmbeddingLayer
from torch.nn import LSTM
from utils import nn_utils, constants
from models.protein_sequence_classification import ProteinSequenceClassification


class LSTM_VirusHostPrediction(ProteinSequenceClassification):
    def __init__(self, vocab_size, n_classes, n_layers, input_dim, hidden_dim, n_mlp_layers):
        super(LSTM_VirusHostPrediction, self).__init__(input_dim, hidden_dim,
                                                       n_mlp_layers=n_mlp_layers,
                                                       n_classes=n_classes,
                                                       batch_norm=False)

        self.embedding = nn.Embedding(vocab_size, input_dim, padding_idx=constants.PAD_TOKEN_VAL)

        # assuming hidden state dimension = cell state dimension = output_dimension = hidden_dim and projection_size=0
        self.lstm = LSTM(input_size=input_dim,
                         hidden_size=hidden_dim,
                         num_layers=n_layers,
                         batch_first=True)

    def get_embedding(self, X):
        X = self.embedding(X.long())
        hidden_input = self.init_zeros(batch_size=X.size(0))
        cell_input = self.init_zeros(batch_size=X.size(0))

        # return values from lstm: output, (hidden_output, cell_output)
        # output: output features from the last layer for each token: num_lstm_layers x batch_size X sequence_length X hidden_dim
        # hidden_output: final hidden state (embedding) for each sequence: num_lstm_layers x batch_size X hidden_dim
        # cell_output: final cell state (embedding) for each sequence: num_lstm_layers x batch_size X hidden_dim
        output, _ = self.lstm(X, (hidden_input, cell_input))

        # aggregate the embeddings from lstm
        # mean of the representations of all tokens
        return output.mean(dim=1)

    # def forward() : use the template implementation in ProteinSequenceClassification

    def init_zeros(self, batch_size):
        # dimensions: N (num of lstm layers) X batch_size X hidden_layer_dimension
        return torch.zeros(self.N, batch_size, self.hidden_dim).to(nn_utils.get_device())


    def get_model(model_params) -> ProteinSequenceClassification:
        model = LSTM_VirusHostPrediction(vocab_size=model_params["vocab_size"],
                                         n_classes=model_params["n_classes"],
                                         n_layers=model_params["n_layers"],
                                         input_dim=model_params["input_dim"],
                                         hidden_dim=model_params["hidden_dim"],
                                         n_mlp_layers=model_params["n_mlp_layers"])

        print(model)
        print("LSTM_VirusHostPredictionN; umber of parameters = ", sum(p.numel() for p in model.parameters() if p.requires_grad))

        return ProteinSequenceClassification.return_model(model, model_params["data_parallel"])
