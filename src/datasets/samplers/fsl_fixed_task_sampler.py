import random
from typing import Iterator, List

import torch
from datasets.samplers.fsl_task_sampler import FewShotLearningTaskSampler


class FewShotLearningFixedTaskSampler(FewShotLearningTaskSampler):
    """
    N-Way-K-Shot Few Shot Learning Sampler
    Samples batches for few-shot classification tasks
    For each batch:
        1. Sample n_way classes from the labels
        2. Sample n_support + n_query samples for each class  
    """

    def __init__(self, dataset, n_way: int, n_shot: int, n_query: int, n_task: int):
        """
        Args:
            dataset: dataset of protein sequences
            n_way: number of classes in one task
            n_shot: number of support sequences in each task
            n_query: number of query sequences in each task
            n_task: number of tasks (a.k.a batches)
        """

        super().__init__(dataset=dataset, n_shot=n_shot, n_query=n_query, n_task=n_task)

        self.n_way = n_way
        # if n_way is not configured, use all the labels in the dataset
        if self.n_way is None:
            self.n_way = self.n_labels

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