import pandas as pd
import torch
import torch_geometric.utils
from torch_geometric.data import Dataset
from os import path
import networkx as nx

from src.utils import utils


class ProteinStructureDataset(Dataset):
    def __init__(self, dirpath, id_filepath):
        super(ProteinStructureDataset, self).__init__()
        self.dirpath = dirpath
        self.id_map = utils.get_id_mapping(id_filepath)
        self.n_prot_ids = None
        self.init_references(dirpath)

    def len(self):
        return len(self.id_map)

    def get(self, idx: int):
        prot_id = self.id_map[idx]
        prot_graph_nx = nx.read_gpickle(path.join(self.dirpath, f"{prot_id}.gpickle"))
        # from_networkx() does not store all the attributes of the node by default
        # hack
        # 1: get list of all attribute keys from the networkx graph and pass it as an argument to from_networkx() to load them
        # 2: attribute keys in the original networkx graph are integers 0, 1, 2, ....
        #          however, torch requires the keys to be strings, so we convert them
        # 3. we have explicitly stored the edge attributes as list in the networkx graph with the var name 'edge_attr'
        # so, we use that var name to read it by passing it as an arg to group_edge_attr
        node_attrs = [str(x) for x in list(prot_graph_nx.nodes[0].keys())]

        prot_graph = torch_geometric.utils.from_networkx(prot_graph_nx, group_node_attrs=node_attrs, group_edge_attrs=["edge_attr"])

        return prot_graph.to(utils.get_device()), label