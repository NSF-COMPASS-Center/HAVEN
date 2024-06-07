from utils import nn_utils
import torch
from sklearn.model_selection import train_test_split


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
    def __init__(self, n_way, n_shot, n_query):
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query

    def __call__(self, batch):
        sequences, labels = zip(*batch)
        support_sequences = []
        support_labels = []
        query_sequences = []
        query_labels = []

        unique_labels = labels.unique(sorted=True)

        # TODO: fix this - incorrect login
        # we need N-way-K-shot split for support and query
        support_sequences, support_labels, query_sequences, query_labels = train_test_split(sequences, labels,
                                                                                           train_size=n_way*n_shot,
                                                                                           test_size=n_way*n_query,
                                                                                           stratify=labels)

