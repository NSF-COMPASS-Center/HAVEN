import random
from typing import Iterator, List

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
        self.label_item_index_map = {} # label: [list of indices of samples of  the label in the dataset]
        self.initialize_label_item_index_map()


    def initialize_label_item_index_map(self):
        print("Initializing label index map for Few Shot Learning train and validate dataset sampler.")
        for index, label in enumerate(self.dataset.get_labels()):
            if label in self.label_item_index_map:
                # if the label is already there in the map, add item (index) to the label's list of indices
                self.label_item_index_map[label].append(index)
            else:
                # label is not present in the map, i.e., new label encountered
                # initialize a list containing the item (index)
                self.label_item_index_map[label] = [index]

        # remove labels without atleast n_shot + n_query samples
        self.label_item_index_map = dict(
            filter(lambda x: len(x[1]) >= self.n_shot + self.n_query, self.label_item_index_map.items())  # x is a tuple key, val from self.label_item_index_map.items()
        )

    def __len__(self):
        return self.n_task

    def __iter__(self) -> Iterator[List[int]]:
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
            labels = random.sample(list(self.label_item_index_map.keys()), self.n_way)
            for label in labels:
                # for each label, randomly sample n_shot + n_query sequences
                sequence_indices.append(torch.tensor(random.sample(self.label_item_index_map[label], self.n_shot + self.n_query)))
            yield torch.cat(sequence_indices).tolist()