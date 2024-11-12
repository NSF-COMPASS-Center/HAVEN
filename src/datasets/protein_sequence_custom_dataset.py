from utils import nn_utils
from datasets.protein_sequence_dataset import ProteinSequenceDataset
from datasets.protein_sequence_with_id_dataset import ProteinSequenceDatasetWithID

import numpy as np
import torch

class ProteinSequenceProstT5Dataset(ProteinSequenceDataset):
    def __init__(self, df, sequence_col, max_seq_len, truncate, label_col):
        super(ProteinSequenceProstT5Dataset, self).__init__(df, sequence_col, label_col, truncate, max_seq_len)

    def __getitem__(self, idx: int):
        # loc selects based on index in df
        # iloc selects based on integer location (0, 1, 2, ...)
        record = self.data.iloc[idx, :]
        sequence = record[self.sequence_col]

        sequence = sequence.replace("U", "X").replace("Z", "X").replace("O", "X")
        # Add spaces between each amino acid for PT5 to correctly use them
        sequence = " ".join(sequence)
        # AAs to 3Di (or if you want to embed AAs): prepend "<AA2fold>"
        # sequence = "<AA2fold>" + " " + sequence

        return sequence, torch.tensor(record[self.label_col], device=nn_utils.get_device())

class ProteinSequenceESM2Dataset(ProteinSequenceDatasetWithID):
    def __init__(self, df, sequence_col, max_seq_len, truncate, label_col, id_col):
        super(ProteinSequenceESM2Dataset, self).__init__(df, id_col, sequence_col, max_seq_len, truncate, label_col)

    def __getitem__(self, idx: int):
        # loc selects based on index in df
        # iloc selects based on integer location (0, 1, 2, ...)
        record = self.data.iloc[idx, :]

        # format the sequence as a tuple with the id as the identifier
        formatted_sequence = (record[self.id_col], record[self.sequence_col])

        return formatted_sequence, torch.tensor(record[self.label_col], device=nn_utils.get_device())