from torch.utils.data import Dataset
from utils import utils, nn_utils, kmer_utils

import pandas as pd
import numpy as np
import torch
import random


class ProteinSequenceKmerDataset(Dataset):
    def __init__(self, df, id_col, sequence_col, label_col, k, kmer_keys):
        super(ProteinSequenceKmerDataset, self).__init__()
        self.sequence_col = sequence_col
        self.label_col = label_col
        self.data = kmer_utils.compute_kmer_features(df, k, id_col, sequence_col, label_col, kmer_keys=kmer_keys)
        self.kmer_keys = list(set(self.data.columns) - set([id_col, self.label_col]))

    def __len__(self) -> int:
        return self.data.shape[0]

    def get_kmer_keys_count(self):
        return len(self.kmer_keys)

    def get_labels(self):
        return self.data[self.label_col]

    def __getitem__(self, idx: int):
        # loc selects based on index in df
        # iloc selects based on integer location (0, 1, 2, ...)
        record = self.data.iloc[idx, :]
        label = record[self.label_col]

        sequence_vector = np.array(record[self.kmer_keys])
        label_vector = np.array([label])

        return torch.tensor(sequence_vector, device=nn_utils.get_device(), dtype=torch.float32), \
               torch.tensor(label_vector, device=nn_utils.get_device()).squeeze()
