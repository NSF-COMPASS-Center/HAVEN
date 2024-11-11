import torch
import torch.nn as nn
from models.baseline.nlp.embedding.embedding import EmbeddingLayer
from torch.nn import RNN
from utils import nn_utils, constants
from models.protein_sequence_classification import ProteinSequenceClassification


class RNN_VirusHostPrediction(ProteinSequenceClassification):
    def __init__(self, vocab_size, n_classes, n_layers, input_dim, hidden_dim, n_mlp_layers):
        super(RNN_VirusHostPrediction, self).__init__(input_dim, hidden_dim,
                                                      n_mlp_layers=n_mlp_layers,
                                                      n_classes=n_classes,
                                                      batch_norm=False)
        self.embedding = nn.Embedding(vocab_size, input_dim, padding_idx=constants.PAD_TOKEN_VAL)

        self.rnn = RNN(input_size=input_dim,
                       hidden_size=hidden_dim,
                       num_layers=n_layers,
                       nonlinearity="tanh",
                       batch_first=True)

    def get_embedding(self, X):
        X = self.embedding(X.long())
        hidden_input = self.init_hidden(batch_size=X.size(0))

        # return values from rnn:
        # output: output features from the last layer for each token: num_rnn_layers x batch_size X sequence_length X hidden_dim
        # hidden_output: final hidden state (embedding) for each sequence: num_rnn_layers x batch_size X hidden_dim
        # the hidden output is essentially the hidden state of the last token of the sequence
        output, hidden_output = self.rnn(X, hidden_input)

        # aggregate the embeddings from rnn
        # pool the model_params embeddings of all tokens in the input sequence using mean
        return output.mean(dim=1)

    # def forward() : use the template implementation in ProteinSequenceClassification

    def init_hidden(self, batch_size):
        # dimensions: N (num of rnn layers) X batch_size X hidden_layer_dimension
        return torch.zeros(self.N, batch_size, self.hidden_dim).to(nn_utils.get_device())


    def get_model(model_params) -> ProteinSequenceClassification:
        model = RNN_VirusHostPrediction(vocab_size=model_params["vocab_size"],
                                        n_classes=model_params["n_classes"],
                                        n_layers=model_params["n_layers"],
                                        input_dim=model_params["input_dim"],
                                        hidden_dim=model_params["hidden_dim"],
                                        n_mlp_layers = model_params["n_mlp_layers"])

        print(model)
        print("RNN_VirusHostPrediction: Number of parameters = ", sum(p.numel() for p in model.parameters() if p.requires_grad))

        return ProteinSequenceClassification.return_model(model, model_params["data_parallel"])
