from torch import nn
import torch.nn.functional as F


class FeedForwardLayer(nn.Module):
    def __init__(self, d, d_ff):
        super(FeedForwardLayer, self).__init__()
        self.W_1 = nn.Linear(d, d_ff)
        self.W_2 = nn.Linear(d_ff, d)

    def forward(self, X):
        Z = F.relu(self.W_1(X))
        return self.W_2(Z)