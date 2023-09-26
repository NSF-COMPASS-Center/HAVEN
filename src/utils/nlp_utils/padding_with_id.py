from utils import nn_utils
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from utils.nlp_utils.padding import Padding


class PaddingWithID(Padding):
    def __init__(self, max_length, pad_value=0):
        super(PaddingWithID, self).__init__(max_length, pad_value)

    def __call__(self, batch):
        ids, sequences, labels = zip(*batch)
        padded_sequences, labels = super(PaddingWithID, self).__call__(list(zip(sequences, labels)))
        return ids, padded_sequences, labels
