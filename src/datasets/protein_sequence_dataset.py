from torch.utils.data import Dataset
from utils import utils, nn_utils, constants
from textwrap import TextWrapper

import numpy as np
import torch


class ProteinSequenceDataset(Dataset):
    def __init__(self, df, sequence_col, label_col, truncate, max_seq_len):
        super(ProteinSequenceDataset, self).__init__()
        self.sequence_col = sequence_col
        self.label_col = label_col
        self.amino_acid_map = constants.AMINO_ACID_VOCABULARY
        self.data = df
        self.max_seq_len = max_seq_len
        if truncate:
            self.data = self.truncate_dataset(df)

    def __len__(self) -> int:
        return self.data.shape[0]

    def truncate_dataset(self, df):
        # Truncating sequences to fixed length of sequence_max_length
        df.loc[:, self.sequence_col] = df[self.sequence_col].apply(lambda x: x[0:self.max_seq_len])
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
        print(type(sequence_vector))
        print(sequence_vector)
        print(type(label_vector))
        print(label_vector)
        return torch.tensor(sequence_vector, device=nn_utils.get_device(), dtype=torch.float64), torch.tensor(label_vector, device=nn_utils.get_device())


