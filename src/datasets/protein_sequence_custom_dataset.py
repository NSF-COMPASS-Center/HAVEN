from utils import nn_utils
from datasets.protein_sequence_dataset import ProteinSequenceDataset

import numpy as np
import torch

class ProteinSequenceProstT5Dataset(ProteinSequenceDataset):
    def __init__(self, df, sequence_col, max_seq_len, truncate, label_col, id_col):
        super(ProteinSequenceProstT5Dataset, self).__init__(df, sequence_col, label_col, truncate, max_seq_len, id_col)

    def __getitem__(self, idx: int):
        # loc selects based on index in df
        # iloc selects based on integer location (0, 1, 2, ...)
        record = self.data.iloc[idx, :]
        sequence = record[self.sequence_col]
        label = record[self.label_col]

        sequence = sequence.replace("U", "X").replace("Z", "X").replace("O", "X")
        # Add spaces between each amino acid for PT5 to correctly use them
        sequence = " ".join(sequence)
        # AAs to 3Di (or if you want to embed AAs): prepend "<AA2fold>"
        # sequence = "<AA2fold>" + " " + sequence

        return sequence, torch.tensor(label, device=nn_utils.get_device())

class ProteinSequenceESM2Dataset(ProteinSequenceDataset):
    def __init__(self, df, sequence_col, max_seq_len, truncate, label_col, id_col):
        super(ProteinSequenceESM2Dataset, self).__init__(df, sequence_col, label_col, truncate, max_seq_len, id_col)

    def __getitem__(self, idx: int):
        # loc selects based on index in df
        # iloc selects based on integer location (0, 1, 2, ...)
        record = self.data.iloc[idx, :]
        sequence = record[self.sequence_col]
        label = record[self.label_col]

        # format the sequence as a tuple with the id as the identifier
        formatted_sequence = (record[self.id_col], sequence)

        return formatted_sequence, torch.tensor(label, device=nn_utils.get_device())