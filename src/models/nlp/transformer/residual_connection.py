import torch.nn as nn
from models.nlp.transformer.layer_normalization import LayerNormalization


class ResidualConnectionLayer(nn.Module):
    def __init__(self):
        super(ResidualConnectionLayer, self).__init__()
        self.layer_norm = LayerNormalization()

    def forward(self, X, sublayer):
        # sublayer: could be a layer of any block - e.g. multihead-attention, feed-forward, rnn, cnn
        # add the original input back to the output of the sublayer below the residual connection
        X = X + sublayer(X)
        # normalize the result
        return self.layer_norm(X)