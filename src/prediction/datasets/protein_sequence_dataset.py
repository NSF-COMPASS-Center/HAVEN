from torch.utils.data import Dataset
from utils import utils, nn_utils

import pandas as pd
import numpy as np
import torch
import random


class ProteinSequenceDataset(Dataset):
    def __init__(self, filepath, sequence_col, max_seq_len, label_settings, classification_type):
        super(ProteinSequenceDataset, self).__init__()
        self.sequence_col = sequence_col
        self.max_seq_len = max_seq_len # TODO: remove this field if positional encoding with convolution works
        self.label_col = label_settings["label_col"]
        self.label_settings = label_settings
        self.classification_type = classification_type
        self.amino_acid_map = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5,
                               'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
                               'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15,
                               'O': 16, 'S': 17, 'U': 18, 'T': 19, 'W': 20,
                               'Y': 21, 'V': 22, 'B': 23, 'Z': 24, 'X': 25,
                               'J': 26}
        self.index_label_map = None
        self.data = self.read_dataset(filepath)
        self.transform_labels()

    def __len__(self) -> int:
        return self.data.shape[0]

    def transform_labels(self):
        self.data, self.index_label_map = utils.transform_labels(self.data, self.classification_type, self.label_settings)

    def read_dataset(self, filepath):
        df = pd.read_csv(filepath, usecols=[self.sequence_col, self.label_col])
        print(f"Read dataset from {filepath}, size = {df.shape}")
        return df

    def __getitem__(self, idx: int):
        record = self.data.loc[idx, :]
        sequence = record[self.sequence_col]
        label = record[self.label_col]

        sequence_vector = np.array([self.amino_acid_map[a] for a in sequence])
        seq_len = len(sequence_vector)
        if seq_len > self.max_seq_len:
            # select a random sub-string
            start_index = random.randint(0, seq_len - self.max_seq_len)
            end_index = start_index + self.max_seq_len
            sequence_vector = sequence_vector[start_index:end_index]
        label_vector = np.array([label])

        return torch.tensor(sequence_vector, device=nn_utils.get_device()), torch.tensor(label_vector, device=nn_utils.get_device())
