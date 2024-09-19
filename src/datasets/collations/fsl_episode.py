from utils import utils, nn_utils, constants
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
    def __init__(self, n_shot, n_query, max_seq_length):
        self.n_shot = n_shot
        self.n_query = n_query
        self.pad_value = constants.PAD_TOKEN_VAL
        self.max_seq_length = max_seq_length

    def __call__(self, batch):
        sequences, labels = zip(*batch)

        # pad sequences
        sequences = [seq.clone().detach() for seq in sequences]
        padded_sequences = utils.pad_sequences(sequences, self.max_seq_length, self.pad_value)

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

        label_indices_map = {}
        # map of label: indices where the label occurs
        for label in idx_label_map.keys():
            label_indices_map[label] = torch.nonzero(labels == label)

        # in the next step we select the first n_shot samples for the support set
        # to induce stochasticity, shuffle the indices for each label so that different samples are chosen as support sequences in each batch
        for key, val in label_indices_map.items():
            shuffled_idx = torch.randperm(val.nelement()) # val is a [n, 1] 2D tensor where n is the number indices (samples) for the corresponding label in key
            label_indices_map[key] = val.view(-1)[shuffled_idx].view(val.shape) # flatten val, shuffle the tensore, reshape back to original size

        # for each label, select the first n_shot samples for support set
        support_indices = torch.cat([val[: self.n_shot] for val in label_indices_map.values()]).flatten()

        # for each label, select the remaining samples (i.e., after the n_shot support samples) for query set.
        # this allows for varying query set size for each label and can be used for the test dataset loader as is.
        query_indices = torch.cat([val[self.n_shot:] for val in label_indices_map.values()]).flatten()

        support_sequences = padded_sequences[support_indices]
        support_labels = labels[support_indices]
        query_sequences = padded_sequences[query_indices]
        query_labels = labels[query_indices]

        return support_sequences, support_labels, query_sequences, query_labels, idx_label_map
