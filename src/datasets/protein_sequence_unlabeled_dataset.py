from torch.utils.data import Dataset
from utils import utils, nn_utils, constants
from textwrap import TextWrapper

import numpy as np
import torch
from utils import constants
from datasets.protein_sequence_dataset import ProteinSequenceDataset

class ProteinSequenceUnlabeledDataset(ProteinSequenceDataset):
    def __init__(self, df, sequence_col, max_seq_len, truncate, split_sequence, cls_token):
        super(ProteinSequenceUnlabeledDataset, self).__init__(df=df, sequence_col=sequence_col, max_seq_len=max_seq_len,
                                                              truncate=truncate, label_col=None)
        self.cls_token = cls_token
        if split_sequence:
            self.split_sequences()

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
        if self.cls_token:
            # add the CLS token at the beginning of the sequence
            sequence_vector = np.insert(sequence_vector, 0, constants.CLS_TOKEN_VAL)
        return torch.tensor(sequence_vector, device=nn_utils.get_device(), dtype=torch.float64)
