import torch

from datasets.collations.padding import Padding
import numpy as np

class ProteomeCollation(Padding):
    def __init__(self, max_seq_length):
        super(ProteomeCollation, self).__init__(max_seq_length)

    def __call__(self, batch):
        ids, sequences, labels = zip(*batch)
        ids = np.array(list(ids))
        proteome_ids = []
        proteomes = []
        proteome_labels = []
        for p_id in list(set(ids)):
            indices = np.where(ids == p_id)[0]
            p_sequences = [sequences[i] for i in indices]
            p_labels = [labels[i] for i in indices]
            padded_sequences, p_labels = super(ProteomeCollation, self).__call__(list(zip(p_sequences, p_labels)))
            proteome_ids.append(p_id)
            proteomes.append(padded_sequences)
            proteome_labels.append(p_labels[0])
        return proteome_ids, proteomes, proteome_labels
