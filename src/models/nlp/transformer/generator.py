from torch import nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, d, vocab_size):
        super(Generator, self).__init__()
        # project from model dimension to the number of tokens in the vocabulary to predict the next token
        self.linear_projection = nn.Linear(d, vocab_size)

    def forward(self, X):
        return F.log_softmax(self.linear_projection(X), dim=-1)