import torch
import torch.nn as nn
import torch.nn.functional as F
from models.nlp.embedding.embedding import EmbeddingLayer
from torch.nn import Conv1d
from models.protein_sequence_classification import ProteinSequenceClassification
from utils import nn_utils, constants


class CNN_1D_VirusHostPrediction(ProteinSequenceClassification):
    def __init__(self, vocab_size, n_classes, n_layers, n_mlp_layers, input_dim, hidden_dim, kernel_size, stride):
        super(CNN_1D_VirusHostPrediction, self).__init__(input_dim, hidden_dim,
                                                         n_mlp_layers=n_mlp_layers,
                                                         n_classes=n_classes,
                                                         batch_norm=False)
        self.embedding = nn.Embedding(vocab_size, input_dim, padding_idx=constants.PAD_TOKEN_VAL)
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
        # intermediate hidden layers (number = n_layers-1): hidden_dim --> hidden_dim
        # N-1 because we already have one layer converting input_dim --> hidden_dim
        self.conv1d_hidden_layers = nn_utils.create_clones(self.conv1d_hidden, n_layers-1)

    def get_embedding(self, X):
        X = self.embedding(X.long())  # b x n x d
        X = torch.einsum("bnd->bdn", X)  # b x d x n (conv1d requires number of channels as the second dimension)
        X = self.conv1d(X)
        X = torch.einsum("bdn -> bnd", X)  # revert back to b x n x d (batch x updated length of sequence x output_channels dimension)
        X = F.relu(X)  # activation function

        for conv1d_hidden_layer in self.conv1d_hidden_layers:
            X = torch.einsum("bnd->bdn", X)  # b x d x n (conv1d requires number of channels as the second dimension)
            X = conv1d_hidden_layer(X)
            X = torch.einsum("bdn -> bnd", X)  # revert back to b x n x d (batch x updated length of sequence x output_channels dimension)
            X = F.relu(X)  # activation function

        # pool the model_params embeddings of all tokens in the input sequence using mean
        return X.mean(dim=1)

    # def forward() : use the template implementation in ProteinSequenceClassification

    def get_model(model_params) -> CNN_1D_VirusHostPrediction:
        model = CNN_1D_VirusHostPrediction(vocab_size=model_params["vocab_size"],
                                           n_classes=model_params["n_classes"],
                                           n_layers = model_params["n_layers"],
                                           n_mlp_layers = model_params["n_mlp_layers"],
                                           input_dim= model_params["input_dim"],
                                           hidden_dim= model_params["hidden_dim"],
                                           kernel_size= model_params["kernel_size"],
                                           stride=model_params["stride"])

        print(model)
        print("CNN_1D_VirusHostPrediction: Number of parameters = ", sum(p.numel() for p in model.parameters() if p.requires_grad))

        return VirusHostPredictionBase.return_model(model, model_params["data_parallel"])
