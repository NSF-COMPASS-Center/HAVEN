import torch.nn as nn


class NormalizationLayer(nn.Module):
    def __init__(self, eps=1e-6):
        super(NormalizationLayer, self).__init__()
        self.eps = eps

    def forward(self, X):
        mean = X.mean(-1, keepdim=True)
        std_dev = X.std(-1, keepdim=True)
        return (X - mean) / (std_dev + self.eps)