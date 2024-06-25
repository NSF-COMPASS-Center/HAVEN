import random
from typing import Iterator, List
from abc import abstractmethod
import torch
from torch.utils.data import Sampler


class FewShotLearningTaskSampler(Sampler):
    """
    Template parent class for N-Way-K-Shot Few Shot Learning Sampler
    Samples batches for few-shot classification tasks
    For each batch:
        1. Sample n_way classes from the labels
        2. Sample n_support + n_query samples for each class  
    """

    def __init__(self, dataset, n_shot, n_query, n_task):
        """
        Args:
            dataset: dataset of protein sequences
            n_task: number of tasks (a.k.a batches)
        """

        super().__init__(data_source=None)
        self.dataset = dataset
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
        self.filter_labels()
        self.n_labels = len(self.label_item_index_map) # total number of labels eligible for few shot learning

    def filter_labels(self):
        # remove labels without atleast n_shot + n_query samples
        self.label_item_index_map = dict(
            filter(lambda x: len(x[1]) >= self.n_shot + self.n_query, self.label_item_index_map.items())
            # x is a tuple key, val from self.label_item_index_map.items()
        )

    def __len__(self):
        return self.n_task

    @abstractmethod
    def __iter__(self) -> Iterator[List[int]]:
        """
        Return:
            list of indices of length n_way * (n_shot + n_query)
        """
        raise NotImplementedError