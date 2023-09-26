import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d
from utils import nn_utils


class CNN_2D_Model(nn.Module):
    def __init__(self, n_classes, N, n_filters, kernel_size, stride, img_size):
        super(CNN_2D_Model, self).__init__()
        # padding: same ensures the output has the same size as the input
        self.conv2d = Conv2d(in_channels=1,
                             out_channels=n_filters,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding="same")
        self.conv2d_hidden = Conv2d(in_channels=n_filters,
                                    out_channels=n_filters,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding="same")
        # intermediate hidden layers (number = N-1): hidden_dim --> hidden_dim
        # N-1 because we already have one layer converting input_dim --> hidden_dim
        self.conv2d_hidden_layers = nn_utils.create_clones(self.conv2d_hidden, N - 1)
        self.linear = nn.Linear(img_size * img_size * n_filters, n_classes)

    def forward(self, X):
        X = F.relu(self.conv2d(X))
        for conv2d_hidden_layer in self.conv2d_hidden_layers:
            X = F.relu(conv2d_hidden_layer(X))

        # aggregate the embeddings from cnn
        # mean of the representations of all tokens
        self.cnn_emb = torch.flatten(X, 1)  # flatten all dimensions except batch
        y = self.linear(self.cnn_emb)
        return y


def get_cnn_model(model):
    cnn_model = CNN_2D_Model(n_classes=model["n_classes"],
                             N=model["depth"],
                             n_filters=model["n_filters"],
                             kernel_size=model["kernel_size"],
                             stride=model["stride"],
                             img_size=model["img_size"])

    print(cnn_model)
    print("Number of parameters = ", sum(p.numel() for p in cnn_model.parameters() if p.requires_grad))
    return cnn_model.to(nn_utils.get_device())
