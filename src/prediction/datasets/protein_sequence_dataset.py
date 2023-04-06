from torch.utils.data import Dataset
from utils import nn_utils

import pandas as pd
import numpy as np
import torch


class ProteinSequenceDataset(Dataset):
    def __init__(self, filepath, sequence_col, label_col, label_classes, sequence_max_length):
        super(ProteinSequenceDataset, self).__init__()
        self.sequence_col = sequence_col
        self.label_col = label_col
        self.label_classes = label_classes
        self.n_label_classes = len(label_classes)
        self.sequence_max_length = sequence_max_length
        self.initialize_references()
        self.data = self.read_dataset(filepath)

    def __len__(self) -> int:
        return self.data.shape[0]

    def initialize_references(self):
        self.amino_acid_map = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5,
                               'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
                               'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15,
                               'O': 16, 'S': 17, 'U': 18, 'T': 19, 'W': 20,
                               'Y': 21, 'V': 22, 'B': 23, 'Z': 24, 'X': 25,
                               'J': 26}

        self.label_index_map = {}
        self.index_label_map = {}
        # for some weird reason, yaml parses the label_classes as tuple (<list>, )
        # with the second part of the tupe as None
        for index, label in enumerate(self.label_classes[0]):
            self.label_index_map[label] = index
            self.index_label_map[index] = label

    def read_dataset(self, filepath):
        df = pd.read_csv(filepath, usecols=[self.sequence_col, self.label_col])
        print(f"Read dataset from {filepath}, size = {df.shape}")
        # Truncating sequences to fixed length of sequence_max_length
        df[self.sequence_col] = df[self.sequence_col].apply(lambda x: x[0:self.sequence_max_length])
        return df

    def __getitem__(self, idx: int):
        record = self.data.loc[idx, :]
        sequence = record[self.sequence_col]
        label = record[self.label_col]

        sequence_vector = np.array([self.amino_acid_map[a] for a in sequence])

        if label not in self.label_index_map:
            label = "Others"

        label_vector = np.array([self.label_index_map[label]])

        return torch.tensor(sequence_vector, device=nn_utils.get_device()), torch.tensor(label_vector, device=nn_utils.get_device())
