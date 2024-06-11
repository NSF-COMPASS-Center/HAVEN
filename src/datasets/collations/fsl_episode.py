from utils import utils, nn_utils
import torch
import numpy as np

class FewShotLearningEpisode:
    """
    Collate function for N-Way-K-Shot Few Shot Learning dataloaders.
    Args:
        batch: (protein_sequence, label)
    Returns:
        tuple (support_sequences, support_labels, query_sequences, query_labels)
            - # support_sequences: n_way * n_shot
            - # support_labels: n_way * n_shot
            - # query_sequences: n_way * n_query
            - # query_labels: n_way * n_query
    """
    def __init__(self, n_way, n_shot, n_query, pad_value, max_length):
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.pad_value = pad_value
        self.max_length = max_length

    def __call__(self, batch):
        sequences, labels = zip(*batch)

        # pad sequences
        sequences = [seq.clone().detach() for seq in sequences]
        padded_sequences = utils.pad_sequences(sequences, self.max_length, self.pad_value)

        support_sequences = []
        support_labels = []
        query_sequences = []
        query_labels = []

        labels = np.array(labels)
        unique_labels = list(set(labels))
        label_idx_map, idx_label_map = utils.get_label_vocabulary(unique_labels)

        # convert the labels to integers
        for key, val in label_idx_map.items():
            labels = np.where(labels == key, val, labels)
        labels = torch.tensor(labels.astype(float), device=nn_utils.get_device())

        support_sequence = []
        support_labels = []

        label_indices = torch.cat([torch.nonzero(labels == label) for label in idx_label_map.keys()])
        label_indices = label_indices.reshape((self.n_way, self.n_shot + self.n_query)) # assuming the the labels are ordered as [[0, 0, 0, ...0, 1, 1, ..., 1, 2, 2, ...., 2]]
        support_indices = label_indices[:, : self.n_shot].flatten()
        query_indices = label_indices[:, self.n_shot : self.n_shot + self.n_query + 1].flatten() # assuming there are n_shot + n_query samples

        support_sequences = padded_sequences[support_indices]
        support_labels = labels[support_indices]
        query_sequences = padded_sequences[query_indices]
        query_labels = labels[query_indices]

        return support_sequences, support_labels, query_sequences, query_labels, idx_label_map
