import torch.nn as nn
import torch.nn.functional as F
from models.nlp.embedding.embedding import EmbeddingLayer
from models.virus_host_prediction_base import VirusHostPredictionBase
from utils import nn_utils, constants


class FNN_VirusHostPrediction(VirusHostPredictionBase):
    def __init__(self, vocab_size, n_classes, n_mlp_layers, input_dim, hidden_dim):
        super(FNN_VirusHostPrediction, self).__init__(input_dim, hidden_dim, n_mlp_layers, n_classes, batch_norm=False)

        self.embedding = nn.Embedding(vocab_size, input_dim, padding_idx=constants.PAD_TOKEN_VAL)

    def get_embedding(self, X):
        X = self.embedding(X.long())
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
    fnn_model = FNN_Model(vocab_size=model["vocab_size"],
                          n_classes=model["n_classes"],
                          N=model["n_mlp_layers"],
                          input_dim=model["input_dim"],
                          hidden_dim=model["hidden_dim"])

    print(fnn_model)
    print("Number of parameters = ", sum(p.numel() for p in fnn_model.parameters() if p.requires_grad))
    return fnn_model.to(nn_utils.get_device())
