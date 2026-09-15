import random
from typing import Iterator, List
from torch.utils.data import Sampler
import torch


class ProteomeSampler(Sampler):
    """
    Proteome Sampler
    Samples batches of proteomes
    For each batch:
        1. Sample batch_size of proteomes
        2. Sample all proteins in each proteome
    """

    def __init__(self, dataset, batch_size, id_col):
        """
        Args:
            dataset: dataset of protein sequences
            batch_size: batch_size
        """
        super().__init__(data_source=None)
        self.dataset = dataset
        self.id_col = id_col
        self.batch_size = batch_size
        self.proteome_ids = self.dataset.data[self.id_col].unique().tolist()

    def __len__(self):
        n_proteomes = len(self.proteome_ids)
        return n_proteomes // self.batch_size

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.__len__()):
            sequence_indices = []
            # 1. randomly sample batch_size proteomes
            sampled_proteome_ids = random.sample(list(self.proteome_ids), self.batch_size)
            for proteome_id in sampled_proteome_ids:
                # for each sampled proteome_ids, select all the corresponding proteins in the proteome.
                sequence_indices.append(torch.tensor(self.dataset.data[self.dataset.data[self.id_col] == proteome_id].index))
            yield torch.cat(sequence_indices).tolist()