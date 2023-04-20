from utils import nn_utils
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


class Padding:
    def __init__(self, max_length, pad_value=0):
        self.pad_value = pad_value
        self.max_length = max_length

    def __call__(self, batch):
        sequences, labels = zip(*batch)
        sequences = [seq.clone().detach() for seq in sequences]
        # pad the first sequence to the desired fixed length
        # NOTE: the fixed length padding will work only if size of all sequences are less than or equal to the desired max_length
        sequences[0] = nn.ConstantPad1d((0, self.max_length - sequences[0].shape[0]), self.pad_value)(sequences[0])

        # use pytorch utility pad_sequences for variable length padding w.r.t to the first sequence
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=self.pad_value)

        return padded_sequences, torch.tensor(labels, device=nn_utils.get_device())
