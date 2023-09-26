import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, MaxPool2d
from utils import nn_utils


class CNN_2D_MaxPool_Model(nn.Module):
    def __init__(self, n_classes, N, n_filters, kernel_size, stride, img_size):
        super(CNN_2D_MaxPool_Model, self).__init__()
        # padding: same ensures the output has the same size as the input
        self.conv2d_1 = Conv2d(in_channels=1,
                               out_channels=int(n_filters / 2),
                               kernel_size=kernel_size,
                               stride=stride,
                               padding="same")
        self.pool = MaxPool2d(2, 2)
        self.conv2d_2 = Conv2d(in_channels=int(n_filters / 2),
                               out_channels=n_filters,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding="same")
        hidden_dim = int(n_filters * img_size / 4 * img_size / 4)
        self.linear_1 = nn.Linear(hidden_dim, int(hidden_dim / 2))
        self.linear_2 = nn.Linear(int(hidden_dim / 2), int(hidden_dim / 4))
        self.linear_3 = nn.Linear(int(hidden_dim / 4), n_classes)

    def forward(self, X):
        X = self.pool(F.relu(self.conv2d_1(X)))
        X = self.pool(F.relu(self.conv2d_2(X)))
        X = torch.flatten(X, 1)  # flatten all dimensions except batch

        X = F.relu(self.linear_1(X))
        self.cnn_emb = F.relu(self.linear_2(X))

        y = self.linear_3(self.cnn_emb)
        return y


def get_cnn_model(model):
    cnn_model = CNN_2D_MaxPool_Model(n_classes=model["n_classes"],
                                     N=model["depth"],
                                     n_filters=model["n_filters"],
                                     kernel_size=model["kernel_size"],
                                     stride=model["stride"],
                                     img_size=model["img_size"])

    print(cnn_model)
    print("Number of parameters = ", sum(p.numel() for p in cnn_model.parameters() if p.requires_grad))
    return cnn_model.to(nn_utils.get_device())
