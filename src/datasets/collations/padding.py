from utils import utils, nn_utils
import torch
import torch.nn as nn

class Padding:
    def __init__(self, max_seq_length, pad_value=0):
        self.pad_value = pad_value
        self.max_length = max_seq_length

    def __call__(self, batch):
        sequences, labels = zip(*batch)
        sequences = [seq.clone().detach() for seq in sequences]
        padded_sequences = utils.pad_sequences(sequences, self.max_seq_length, self.pad_value)

        return padded_sequences, torch.tensor(labels, device=nn_utils.get_device())


class PaddingUnlabeled:
    def __init__(self, max_seq_length, pad_value=0):
        self.pad_value = pad_value
        self.max_seq_length = max_seq_length

    def __call__(self, batch):
        sequences = [seq.clone().detach() for seq in batch]
        padded_sequences = utils.pad_sequences(sequences, self.max_seq_length, self.pad_value)
        return padded_sequences