from torch.utils.data import Dataset
from utils import utils, nn_utils
from textwrap import TextWrapper

import numpy as np
import torch


class ProteinSequenceUnlabeledDataset(Dataset):
    def __init__(self, df, sequence_col, max_seq_len, truncate, split_sequences):
        super(ProteinSequenceUnlabeledDataset, self).__init__()
        self.sequence_col = sequence_col
        self.max_seq_len = max_seq_len
        self.amino_acid_map = {"A": 1, "R": 2, "N": 3, "D": 4, "C": 5,
                               "Q": 6, "E": 7, "G": 8, "H": 9, "I": 10,
                               "L": 11, "K": 12, "M": 13, "F": 14, "P": 15,
                               "O": 16, "S": 17, "U": 18, "T": 19, "W": 20,
                               "Y": 21, "V": 22, "B": 23, "Z": 24, "X": 25,
                               "J": 26, "-": 0}
        self.data = df
        if truncate:
            self.data = self.truncate_dataset(df)
        if split_sequences:
            self.split_sequences()

    def __len__(self) -> int:
        return self.data.shape[0]

    def truncate_dataset(self, df):
        # Truncating sequences to fixed length of sequence_max_length
        df[self.sequence_col] = df[self.sequence_col].apply(lambda x: x[0:self.max_seq_len])
        return df

    def split_sequences(self):
        text_wrappper = TextWrapper(width=self.max_seq_len)
        # decompose the sequence column into a list of strings of broken down sequences
        self.data[self.sequence_col] = self.data[self.sequence_col].apply(lambda x: text_wrappper.wrap(x)) # returns a list of substrings of the desired length
        # explode the sequece column
        self.data = self.data.explode(self.sequence_col)

    def __getitem__(self, idx: int):
        # loc selects based on index in df
        # iloc selects based on integer location (0, 1, 2, ...)
        record = self.data.iloc[idx, :]
        sequence = record[self.sequence_col]
        sequence_vector = np.array([self.amino_acid_map[a] for a in sequence])

        return torch.tensor(sequence_vector, device=nn_utils.get_device(), dtype=torch.float64)
