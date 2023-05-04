import torch
from torch_geometric.data import Dataset

import pandas as pd
import numpy as np

from os import path

from src.utils import utils, nn_utils


class ProteinDataset(Dataset):
    def __init__(self, id_filepath, struct_dirpath, seq_filepath, id_col, sequence_col, max_seq_len, truncate,
                 label_settings):
        super(ProteinDataset, self).__init__()
        self.id_map = self.get_id_mapping(id_filepath)
        # structure data
        self.struct_dirpath = struct_dirpath

        # sequence data
        self.id_col = id_col
        self.sequence_col = sequence_col
        self.max_seq_len = max_seq_len
        self.label_col = label_settings["label_col"]
        self.label_settings = label_settings
        self.amino_acid_map = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5,
                               'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
                               'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15,
                               'O': 16, 'S': 17, 'U': 18, 'T': 19, 'W': 20,
                               'Y': 21, 'V': 22, 'B': 23, 'Z': 24, 'X': 25,
                               'J': 26}
        self.data = self.read_dataset(seq_filepath, truncate)
        self.data, self.seq_index_label_map = utils.transform_labels(self.data, self.label_settings)

    def len(self):
        return len(self.id_map)

    def get(self, idx: int):
        # get the protein id corresponding to the index
        prot_id = self.id_map[idx]

        # sequence data
        record = self.data.loc[prot_id]
        sequence = record[self.sequence_col]
        seq_label = record[self.label_col]

        sequence_vector = np.array([self.amino_acid_map[a] for a in sequence])
        seq_label_vector = np.array([seq_label])

        sequence_tensor = torch.tensor(sequence_vector, device=nn_utils.get_device())
        seq_label_tensor = torch.tensor(seq_label_vector, device=nn_utils.get_device())

        prot_graph_nx, prot_graph = nn_utils.load_nx_graph(path.join(self.struct_dirpath, f"{prot_id}.gpickle"))
        prot_graph = prot_graph.to(nn_utils.get_device())

        graph_label_tensor = torch.tensor(np.array([prot_graph_nx.label]), device=nn_utils.get_device())

        # sequence, sequence_label, graph, graph_label
        return sequence_tensor, seq_label_tensor, prot_graph, graph_label_tensor

    def read_dataset(self, filepath, truncate):
        df = pd.read_csv(filepath, usecols=[self.sequence_col, self.label_col], index_col=self.id_col)
        print(f"Read dataset from {filepath}, size = {df.shape}")
        if truncate:
            # Truncating sequences to fixed length of sequence_max_length
            df[self.sequence_col] = df[self.sequence_col].apply(lambda x: x[0:self.max_seq_len])
        return df

    @staticmethod
    def get_id_mapping(filepath):
        print(f"Reading protein ids from {filepath}")
        prot_ids = None
        prot_ids_map = {}
        with open(filepath, "r") as f:
            prot_ids = f.read().split("\n")

        for idx, prot_id in enumerate(prot_ids):
            prot_ids_map[idx] = prot_id

        return prot_ids_map

