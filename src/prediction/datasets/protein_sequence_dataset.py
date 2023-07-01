from torch.utils.data import Dataset
from utils import utils, nn_utils

import pandas as pd
import numpy as np
import torch
import random


class ProteinSequenceDataset(Dataset):
    def __init__(self, df, sequence_col, max_seq_len, truncate, label_col):
        super(ProteinSequenceDataset, self).__init__()
        self.sequence_col = sequence_col
        self.max_seq_len = max_seq_len
        self.label_col = label_col
        self.amino_acid_map = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5,
                               'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
                               'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15,
                               'O': 16, 'S': 17, 'U': 18, 'T': 19, 'W': 20,
                               'Y': 21, 'V': 22, 'B': 23, 'Z': 24, 'X': 25,
                               'J': 26}
        self.data = df
        if truncate:
            self.data = self.truncate_dataset(df)

    def __len__(self) -> int:
        return self.data.shape[0]

    def truncate_dataset(self, df):
        # Truncating sequences to fixed length of sequence_max_length
        df[self.sequence_col] = df[self.sequence_col].apply(lambda x: x[0:self.max_seq_len])
        return df

    def get_labels(self):
        return self.data[self.label_col]

    def __getitem__(self, idx: int):
        # loc selects based on index in df
        # iloc selects based on integer location (0, 1, 2, ...)
        record = self.data.iloc[idx, :]
        sequence = record[self.sequence_col]
        label = record[self.label_col]

        sequence_vector = np.array([self.amino_acid_map[a] for a in sequence])
        label_vector = np.array([label])

        return torch.tensor(sequence_vector, device=nn_utils.get_device()), torch.tensor(label_vector, device=nn_utils.get_device())
