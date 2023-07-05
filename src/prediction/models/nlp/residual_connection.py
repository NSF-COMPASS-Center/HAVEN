import torch.nn as nn
from prediction.models.nlp.normalization import NormalizationLayer


class ResidualConnectionLayer(nn.Module):
    def __init__(self):
        super(ResidualConnectionLayer, self).__init__()
        self.norm = NormalizationLayer()

    def forward(self, X, sublayer):
        # sublayer: could be a layer of any block - e.g. multihead-attention, feed-forward, rnn, cnn
        # add the original input back to the output of the sublayer below the residual connection
        X = X + sublayer(X)
        # normalize the result
        return self.norm(X)