import random
from typing import Iterator, List

import torch
from datasets.samplers.fsl_task_sampler import FewShotLearningTaskSampler

class FewShotLearningVaryingTaskSampler(FewShotLearningTaskSampler):
    """
    N-Way-K-Shot Few Shot Learning Sampler
    Samples batches for few-shot classification tasks
    For each batch:
        1. Randomly choose the number of labels from n_way_range for each task/batch
        2. Sample n_way classes from the labels
        3. Sample n_support + n_query samples for each class
    """

    def __init__(self, dataset, n_way_range: List[int], n_shot: int, n_query: int, n_task: int):
        """
        Args:
            dataset: dataset of protein sequences
            n_way_range: range from which the number of classes in every task is chosen from
            n_shot: number of support sequences in each task
            n_query: number of query sequences in each task
            n_task: number of tasks (a.k.a batches)
        """

        super().__init__(dataset=dataset, n_shot=n_shot, n_query=n_query, n_task=n_task)

        # configure n_way
        self.n_way_range = n_way_range  # expected list of one or two integers: [n_way_min, ] or [n_way_min, n_way_max]
        if len(self.n_way_range) == 1:
            # if n_way_range is configured as [n_way_min, ]
            # set n_way_max as the total number of labels
            self.n_way_range.append(self.n_labels)

        self.n_way_min = self.n_way_range[0]
        self.n_way_max = self.n_way_range[1]

    def __iter__(self) -> Iterator[List[int]]:
        """
        For each task:
            1. Randomly choose the number of labels (n_way) for each task from the given n_way_range
            2. Sample n_way labels uniformly at random from the set of labels
            3. For each label, sample n_shot + n_query sequences uniformly at random from the dataset
        Return:
            list of indices of length n_way * (n_shot + n_query)
        """

        for _ in range(self.n_task):
            sequence_indices = []
            # for each batch, randomly select the number of labels(n_way)
            n_way = random.randint(a=self.n_way_min, b=self.n_way_max)

            # randomly sample n_way labels
            labels = random.sample(list(self.label_item_index_map.keys()), n_way)
            for label in labels:
                # for each label, randomly sample n_shot + n_query sequences
                sequence_indices.append(torch.tensor(random.sample(self.label_item_index_map[label], self.n_shot + self.n_query)))
            yield torch.cat(sequence_indices).tolist()