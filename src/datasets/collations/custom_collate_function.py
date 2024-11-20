import torch
from utils import nn_utils

class ESM2CollateFunction:
    def __init__(self, include_id_col=False):
        self.include_id_col = include_id_col

    def __call__(self, batch):
        # zip(*batch) returns
        # if id_col is included, then ids, named_sequences, labels
        # else named_sequences, labels
        if self.include_id_col:
            ids, named_sequences, labels = zip(*batch)
            return ids, named_sequences, torch.tensor(labels, device=nn_utils.get_device())
        else:
            named_sequences, labels = zip(*batch)
            return named_sequences, torch.tensor(labels, device=nn_utils.get_device())