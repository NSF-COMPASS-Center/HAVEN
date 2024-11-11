import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_seq_len):
        super(PositionalEncoding, self).__init__()
        # max_seq_len is the maximum length of a sequence in the entire dataset
        positions = torch.arange(max_seq_len).unsqueeze(1)  # max_len X 1
        # k: 0 to d in steps of 2 because
        # range of i is  0, 1, ...., 2k, 2k+1, ..., d-1, d
        k = torch.arange(0, d, 2)  # size=d/2
        # 1000^(-2k/d) = exp(log(1000^(-2k/d))) = exp(-2k/d * log(10000))
        # we drop the multiplicative factor 2 (if not, the sinusoidal wave has very large wavelength) , but why? (why diff from the paper?)
        div_terms = torch.exp(k / d * -math.log(10000.0))  # size=d/2
        pos_enc = torch.zeros(1, max_seq_len, d)
        pos_enc[0, :, 0::2] = torch.sin(positions * div_terms)
        pos_enc[0, :, 1::2] = torch.cos(positions * div_terms)
        self.register_buffer("pos_enc", pos_enc)

    def forward(self, X):
        """
        :param X: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        :return: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        return X + self.pos_enc[0, :X.size(1)]
