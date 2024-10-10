import random
from typing import Iterator, List

import torch
from torch.utils.data import Sampler
from datasets.samplers.fsl_task_sampler import FewShotLearningTaskSampler


class FewShotLearningEvaluateTaskSampler(FewShotLearningTaskSampler):
    """
    N-Way-K-Shot Few Shot Learning Sampler
    Samples batches for few-shot evaluation tasks

    Note: In evaluation tasks, the source of dataset for the support and query sequences are separate.
    They are distinguished by a column named 'dataset_type' with values: 'support', 'query'
    For each batch:
        1. Sample n_way classes from the labels
        2. For each class -
            2.1. Sample n_support samples from the support dataset
            2.2. Sample n_query samples from the query dataset
    """

    def __init__(self, dataset, n_way, n_shot, n_query, n_task):
        """
        Args:
            dataset: dataset of protein sequences
            n_way: number of classes in one task
            n_shot: number of support sequences in each task
            n_query: =-1, no limit on the number of sequences for query set. Use all remaining examples after selecting for support set.
            n_task: number of tasks (a.k.a batches)
        """
        # custom index maps for support and query datasets
        # Need to initialize them before self.initialize_label_item_index_map() is called from the parent class constructor

        self.support_label_index_map = {}  # label: [list of indices of samples of the label in the support dataset]
        self.query_label_index_map = {}  # label: [list of indices of samples of the label in the query dataset]

        super().__init__(dataset=dataset, n_shot=n_shot, n_query=n_query, n_task=n_task)

        self.n_way = n_way
        # if n_way is not configured, use all the labels in the dataset
        if self.n_way is None:
            self.n_way = self.n_labels

    def initialize_label_item_index_map(self):
        print("Initializing label index map for Few Shot Evaluation dataset sampler.")

        self.initialize_label_index_map("support", self.support_label_index_map)
        self.initialize_label_index_map("query", self.query_label_index_map)

        self.filter_labels()

        # labels = union of all labels in the support set and in the query set
        self.labels = list(set(self.support_label_index_map.keys()).union(set(self.query_label_index_map.keys())))
        self.n_labels = len(self.labels) # total number of labels in the query dataset eligible for few shot learning

    def initialize_label_index_map(self, dataset_type, label_index_map):
        print(f"Initializing {dataset_type} label index map for Few Shot Evaluation dataset sampler.")
        # select the appropriate dataset
        labels = self.dataset.data[self.dataset.data["dataset_type"] == dataset_type][self.dataset.label_col]

        for index, label in labels.items(): # using items because we want to retain the index of the sample in the dataset. Using enumerate will reset from 0, 1, 2, ...
            if label in label_index_map:
                # if the label is already there in the map, add item (index) to the label's list of indices
                label_index_map[label].append(index)
            else:
                # label is not present in the map, i.e., new label encountered
                # initialize a list containing the item (index)
                label_index_map[label] = [index]

    def filter_labels(self):
        print(f"Number of labels in the support dataset = {len(self.support_label_index_map)}")
        print(f"Number of labels in the query dataset = {len(self.query_label_index_map)}")

        # remove labels without atleast n_shot samples in the support_label_index_map
        self.support_label_index_map = dict(
            filter(lambda x: len(x[1]) >= self.n_shot, self.support_label_index_map.items())  # x is a tuple (key, val) from self.support_label_index_map.items()
        )
        print(f"Number of labels in the support dataset after filter for atleast n_shot samples = {len(self.support_label_index_map)}")
        filter_out_labels = []
        for key, val in self.query_label_index_map.items():
            if key not in self.support_label_index_map: # key is the label
                # new label: does not have support sequences in the support dataset
                if len(val) < (self.n_shot + 1): # val is the list of indices. We need atleast n_shot + 1 (one for query)
                    # remove the label
                    filter_out_labels.append(key)
        self.query_label_index_map = dict(
            filter(lambda x: x[0] not in filter_out_labels, self.query_label_index_map.items())
        )
        print(f"Number of labels in the query dataset after filter for atleast n_shot samples = {len(self.query_label_index_map)}")
        # Query set: Filter only the new labels in the query set, i.e. labels which are not in the support set, to have minimum n_shots

        # For example: unseen hosts in an unseen virus

    def __iter__(self) -> Iterator[List[int]]:
        """
        1. Sample n_way classes from the labels uniformly at random
        2. For each class -
            2.1. Sample n_support samples from the support dataset
            2.2. Sample n_query samples from the query dataset
        Return:
            list of varying length lists of indices = n_way * (number of sequences)
        """
        for _ in range(self.n_task):
            sequence_indices = []

            # for each batch, randomly sample n_way labels
            labels = random.sample(self.labels, self.n_way)
            for label in labels:
                label_sequence_indices = []
                # using extend to flatten out and avoid nested lists
                if label in self.support_label_index_map:
                    # if the label is present in the support dataset, randomly samples n_shot sequences from the support dataset
                    # this may not always be True because there may be new labels in the query set which are not in the support set
                    # For example: unseen hosts in an unseen virus
                    # In such cases, the support sequences will also be picked from the query dataset
                    label_sequence_indices.extend(random.sample(list(self.support_label_index_map[label]), self.n_shot))
                if label in self.query_label_index_map:
                    # if the label has sequences in the query dataset, select all sequences from the query dataset
                    # this is not necessarity true since all labels will not be present in the query dataset
                    # For example: when the evluation is on unseen hosts, the seen hosts will not be present in the query dataset
                    label_sequence_indices.extend(list(self.query_label_index_map[label]))

                sequence_indices.append(torch.tensor(label_sequence_indices))
            yield torch.cat(sequence_indices).tolist()