from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import copy
import os

from prediction.datasets.protein_sequence_dataset import ProteinSequenceDataset
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


def get_dataset_loader(input_dir, input, sequence_settings, label_settings, dataset_type=None):
    seq_col = sequence_settings["sequence_col"]
    batch_size = sequence_settings["batch_size"]
    max_seq_len = sequence_settings["max_sequence_length"]
    pad_sequence_val = sequence_settings["pad_sequence_val"]
    truncate = sequence_settings["truncate"]
    # TODO: add support for multiple files in the list. Current implementation supports only one train and one test file.
    filepath = os.path.join(input_dir, input["dir"], input[dataset_type][0])
    dataset = ProteinSequenceDataset(filepath, seq_col, max_seq_len, truncate, label_settings)
    return dataset.index_label_map, DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=Padding(max_seq_len, pad_sequence_val))


def get_criterion(loss):
    criterion = nn.CrossEntropyLoss() # default
    if loss == "MultiMarginLoss":
        criterion = nn.MultiMarginLoss()
    return criterion
