from utils import nn_utils
from datasets.protein_sequence_dataset import ProteinSequenceDataset

import numpy as np
import torch


class ProteinSequenceDatasetWithID(ProteinSequenceDataset):
    def __init__(self, df, id_col, sequence_col, max_seq_len, truncate, label_col):
        super(ProteinSequenceDatasetWithID, self).__init__(df, sequence_col, label_col, truncate, max_seq_len)
        self.id_col = id_col

    def __getitem__(self, idx: int):
        # loc selects based on index in df
        # iloc selects based on integer location (0, 1, 2, ...)
        record = self.data.iloc[idx, :]
        id = record[self.id_col]
        sequence = record[self.sequence_col]
        label = record[self.label_col]

        id_vector = np.array([id])
        sequence_vector = np.array([self.amino_acid_map[a] for a in sequence])
        label_vector = np.array([label])

        return id_vector, \
               torch.tensor(sequence_vector, device=nn_utils.get_device(), dtype=torch.float64), \
               torch.tensor(label_vector, device=nn_utils.get_device())


