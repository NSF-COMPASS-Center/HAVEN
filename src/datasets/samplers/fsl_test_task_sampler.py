import random
from typing import Iterator, List

import torch
from torch.utils.data import Sampler
from datasets.samplers.fsl_task_sampler import FewShotLearningTaskSampler


class FewShotLearningTestTaskSampler(FewShotLearningTaskSampler):
    """
    N-Way-K-Shot Few Shot Learning Sampler
    Samples batches for few-shot classification tasks
    For each batch:
        1. Sample n_way classes from the labels
        2. Sample n_support + n_query samples for each class  
    """

    def __init__(self, dataset, n_way, n_shot, n_task):
        """
        Args:
            dataset: dataset of protein sequences
            n_way: number of classes in one task
            n_shot: number of support sequences in each task
            n_query: =-1, no limit on the number of sequences for query set. Use all remaining examples after selecting for support set.
            n_task: number of tasks (a.k.a batches)
        """
        super().__init__(dataset=dataset, n_shot=n_shot, n_query=-1, n_task=n_task)

        self.n_way = n_way
        # if n_way is not configured, use all the labels in the dataset
        if self.n_way is None:
            self.n_way = self.n_labels


    def filter_labels(self):
        # remove labels without atleast n_shot + 1 samples
        # + 1 because we need atleast one sample of the label for testing in the query set
        self.label_item_index_map = dict(
            filter(lambda x: len(x[1]) >= self.n_shot + 1, self.label_item_index_map.items())  # x is a tuple (key, val) from self.label_item_index_map.items()
        )

    def __iter__(self) -> Iterator[List[int]]:
        """
        For each task:
            1. Sample n_way labels uniformly at random from the set of labels
            2. For each label, select all its corresponding sequences from the dataset (# of sequences >= n_support + 1)
        Return:
            list of varying length lists of indices = n_way * (number of sequences)
        """
        for _ in range(self.n_task):
            sequence_indices = []
            # for each batch, randomly sample n_way labels
            labels = random.sample(list(self.label_item_index_map.keys()), self.n_way)
            for label in labels:
                # for each label, select all sequences
                # n_shot will be used for support set and the remanining for query set
                sequence_indices.append(torch.tensor(self.label_item_index_map[label]))
            yield torch.cat(sequence_indices).tolist()