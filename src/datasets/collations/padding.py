from utils import utils, nn_utils, constants
import torch
import torch.nn as nn

class Padding:
    def __init__(self, max_seq_length):
        self.pad_value = constants.PAD_TOKEN_VAL
        self.max_seq_length = max_seq_length

    def __call__(self, batch):
        sequences, labels = zip(*batch)
        sequences = [seq.clone().detach() for seq in sequences]
        padded_sequences = utils.pad_sequences(sequences, self.max_seq_length, self.pad_value)

        return padded_sequences, torch.tensor(labels, device=nn_utils.get_device())


class PaddingUnlabeled:
    def __init__(self, max_seq_length, cls_token):
        self.pad_value = constants.PAD_TOKEN_VAL
        if cls_token:
            # self.max_seq_length + 1: adding 1 to account for CLS token which has been added in the ProteinSequenceUnlabeledDataset itself
            # this is not needed in Padding because the CLS token is not added in ProteinSequenceDataset, but in HAVEN.forward() method
            self.max_seq_length = max_seq_length + 1
        else:
            self.max_seq_length = max_seq_length

    def __call__(self, batch):
        sequences = [seq.clone().detach() for seq in batch]
        padded_sequences = utils.pad_sequences(sequences, self.max_seq_length, self.pad_value)
        return padded_sequences