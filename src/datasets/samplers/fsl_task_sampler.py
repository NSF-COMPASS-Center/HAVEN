import random

import torch
from torch.utils.data import Sampler


class FewShotLearningTaskSampler(Sampler):
    """
    N-Way-K-Shot Few Shot Learning Sampler
    Samples batches for few-shot classification tasks
    For each batch:
        1. Sample n_way classes from the labels
        2. Sample n_support + n_query samples for each class  
    """

    def __init__(self, dataset, n_way, n_shot, n_query, n_task):
        """
        Args:
            dataset: dataset of protein sequences
            n_way: number of classes in one task
            n_shot: number of support sequences in each task
            n_query: number of query sequences in each task
            n_task: number of tasks (a.k.a batches)
        """

        super().__init__(data_source=None)
        self.dataset = dataset
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.n_task = n_task
        self.label_index_map: {} # label: [list of indices of samples of  the label in the dataset]
        self.initialize_label_index_map()


    def initialize_label_index_map(self):
        for index, label in enumerate(self.dataset.get_labels()):
            if label in self.label_index_map:
                # if the label is already there in the map, add item (index) to the list of indices
                self.label_index_map.append(index)
            else:
                # label is not present in the map, i.e., new label encountered
                # initialize a list containing the item (index)
                self.label_index_map[label] = [index]

    def __len__(self):
        return self.n_tasks

    def __iter__(self):
        """
        For each task:
            1. Sample n_way labels uniformly at random from the set of labels
            2. For each label, sample n_shot + n_query sequences uniformly at random from the dataset
        Return:
            list of indices of length n_way * (n_shot + n_query)
        """

        for _ in range(self.n_task):
            sequence_indices = []
            # for each batch, randomly sample n_way labels
            labels = random.sample(self.label_index_map.keys(), self.n_way)
            for label in labels:
                # for each label, randomly sample n_shot + n_query sequences
                sequence_indices.append(torch.tensor(random.sample(self.label_index_map[label], self.n_shot + self.n_query)))
            yield torch.cat(sequence_indices).tolist()