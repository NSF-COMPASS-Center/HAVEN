from utils import nn_utils
from datasets.protein_sequence_dataset import ProteinSequenceDataset

import numpy as np
import torch

class ProteinSequenceWithLabelDataset(ProteinSequenceDataset):
    def __init__(self, df, sequence_col, max_seq_len, truncate, label_col):
        super(ProteinSequenceWithLabelDataset, self).__init__(df, sequence_col, max_seq_len, truncate, label_col)

    def __getitem__(self, idx: int):
        # loc selects based on index in df
        # iloc selects based on integer location (0, 1, 2, ...)
        record = self.data.iloc[idx, :]
        sequence = record[self.sequence_col]
        label = record[self.label_col]

        sequence_vector = np.array([self.amino_acid_map[a] for a in sequence])

        return torch.tensor(sequence_vector, device=nn_utils.get_device(), dtype=torch.float64), label