import torch

class ESM2CollateFunction:
    def __call__(self, batch):
        # zip(*batch) returns
        # if id_col is included, then ids, named_sequences, labels
        # else named_sequences, labels
        return zip(*batch)