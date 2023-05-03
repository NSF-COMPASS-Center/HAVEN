from torch.utils.data import Dataset
from utils import utils, nn_utils

import pandas as pd
import numpy as np
import torch


class ProteinSequenceDataset(Dataset):
    def __init__(self, filepath, id_col, sequence_col, max_seq_len, truncate, label_settings, id_filepath):
        super(ProteinSequenceDataset, self).__init__()
        self.id_col = id_col
        self.sequence_col = sequence_col
        self.max_seq_len = max_seq_len
        self.label_col = label_settings["label_col"]
        self.label_settings = label_settings
        self.amino_acid_map = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5,
                               'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
                               'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15,
                               'O': 16, 'S': 17, 'U': 18, 'T': 19, 'W': 20,
                               'Y': 21, 'V': 22, 'B': 23, 'Z': 24, 'X': 25,
                               'J': 26}
        self.data = self.read_dataset(filepath, truncate)
        self.data, self.index_label_map = utils.transform_labels(self.data, self.label_settings)
        self.id_map = utils.get_id_mapping(id_filepath)

    def __len__(self) -> int:
        n = self.data.shape[0]
        if n != len(self.id_map):
            print("ERROR: Number of sequences different from number of ids.")
            return None
        return n

    def read_dataset(self, filepath, truncate):
        df = pd.read_csv(filepath, usecols=[self.sequence_col, self.label_col], index_col=self.id_col)
        print(f"Read dataset from {filepath}, size = {df.shape}")
        if truncate:
            # Truncating sequences to fixed length of sequence_max_length
            df[self.sequence_col] = df[self.sequence_col].apply(lambda x: x[0:self.max_seq_len])
        return df

    def __getitem__(self, idx: int):
        record = self.data.loc[self.id_map[idx], :]
        sequence = record[self.sequence_col]
        label = record[self.label_col]

        sequence_vector = np.array([self.amino_acid_map[a] for a in sequence])
        label_vector = np.array([label])

        return torch.tensor(sequence_vector, device=nn_utils.get_device()), torch.tensor(label_vector, device=nn_utils.get_device())
