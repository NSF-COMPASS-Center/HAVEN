from torch.utils.data import DataLoader
import torch_geometric
import torch.nn as nn
import torch
import copy
import os
import networkx as nx

from prediction.datasets.protein_sequence_dataset import ProteinSequenceDataset
from prediction.datasets.protein_dataset import ProteinDataset
from prediction.models.nlp.padding import Padding


def create_clones(module, N):
    """
    Returns create N identical layers of a given neural network module,
    examples of modules: feed-forward, multi-head attention, or even a layer of encoder (which has multiple layers multi-head attention and feed-forward layers within it)
    :param module: neural network module
    :param N: number of clones to be created
    :return: List of N clones of module
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def get_device(tensor=None):
    """
    Returns a device string either for the best available device,
    or for the device corresponding to the argument
    :param tensor:
    :return:
    """
    device = "cpu"
    if tensor is None:
        if torch.cuda.is_available():
            device = "cuda"
    else:
        if tensor.is_cuda:
            device = "cuda"
    return device


def get_protein_sequence_dataset_loader(input_dir, input, sequence_settings, label_settings, dataset_type=None):
    max_seq_len = sequence_settings["max_sequence_length"]
    # TODO: add support for multiple files in the list. Current implementation supports only one train and one test file.
    filepath = os.path.join(input_dir, input["dir"], input[dataset_type][0])

    dataset = ProteinSequenceDataset(filepath=filepath,
                                     sequence_col=sequence_settings["sequence_col"],
                                     max_seq_len=max_seq_len,
                                     truncate=sequence_settings["truncate"],
                                     label_settings=label_settings)
    return dataset.index_label_map, DataLoader(dataset=dataset,
                                               batch_size=sequence_settings["batch_size"],
                                               shuffle=True,
                                               collate_fn=Padding(max_seq_len, sequence_settings["pad_sequence_val"]))


def get_protein_dataset_loader(id_filepath, struct_input_dir, seq_input_dir, input, sequence_settings, label_settings,
                               dataset_type=None):
    max_seq_len = sequence_settings["max_sequence_length"]
    # TODO: add support for multiple files in the list. Current implementation supports only one train and one test file.
    seq_filepath = os.path.join(seq_input_dir, input["dir"], input[dataset_type][0])

    dataset = ProteinDataset(id_filepath=id_filepath.format(dataset_type=dataset_type),
                             struct_dirpath=struct_input_dir,
                             seq_filepath=seq_filepath,
                             id_col=sequence_settings["id_col"],
                             sequence_col=sequence_settings["sequence_col"],
                             max_seq_len=max_seq_len,
                             truncate=sequence_settings["truncate"],
                             label_settings=label_settings)
    return dataset.seq_index_label_map, torch_geometric.loader.DataLoader(dataset=dataset,
                                                                          batch_size=sequence_settings["batch_size"],
                                                                          shuffle=True,
                                                                          collate_fn=Padding(max_seq_len, sequence_settings["pad_sequence_val"]))


def get_criterion(loss):
    criterion = nn.CrossEntropyLoss()  # default
    if loss == "MultiMarginLoss":
        criterion = nn.MultiMarginLoss()
    return criterion


def load_nx_graph(graph_filepath):
    prot_graph_nx = nx.read_gpickle(graph_filepath)
    # from_networkx() does not store all the attributes of the node by default
    # hack
    # 1: get list of all attribute keys from the networkx graph and pass it as an argument to from_networkx() to load them
    # 2: attribute keys in the original networkx graph are integers 0, 1, 2, ....
    #          however, torch requires the keys to be strings, so we convert them
    # 3. we have explicitly stored the edge attributes as list in the networkx graph with the var name 'edge_attr'
    # so, we use that var name to read it by passing it as an arg to group_edge_attr
    node_attrs = [str(x) for x in list(prot_graph_nx.nodes[0].keys())]

    return prot_graph_nx, torch_geometric.utils.from_networkx(prot_graph_nx, group_node_attrs=node_attrs,
                                                              group_edge_attrs=["edge_attr"])
