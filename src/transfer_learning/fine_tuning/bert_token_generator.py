from torch import nn
import torch.nn.functional as F


class TokenGenerator(nn.Module):
    def __init__(self, d, vocab_size):
        super(TokenGenerator, self).__init__()
        # project from model_params dimension to the number of tokens in the vocabulary to predict the next token
        self.linear_projection = nn.Linear(d, vocab_size)

    def forward(self, X):
        return F.log_softmax(self.linear_projection(X), dim=-1)