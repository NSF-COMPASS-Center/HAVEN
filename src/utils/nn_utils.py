from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import copy

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


def get_dataset_loader(df, sequence_settings, label_col):
    seq_col = sequence_settings["sequence_col"]
    batch_size = sequence_settings["batch_size"]
    max_seq_len = sequence_settings["max_sequence_length"]
    pad_sequence_val = sequence_settings["pad_sequence_val"]
    truncate = sequence_settings["truncate"]
    dataset = ProteinSequenceDataset(df, seq_col, max_seq_len, truncate, label_col)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=Padding(max_seq_len, pad_sequence_val))


def get_criterion(loss):
    criterion = nn.CrossEntropyLoss() # default
    if loss == "MultiMarginLoss":
        criterion = nn.MultiMarginLoss()
    return criterion


def init_weights(module: nn.Module, initialization_type: str, bias_init_value=0):
    try:
        if initialization_type == "uniform":
            # drawn from uniform distribution between 0 and 1
            torch.nn.init.uniform_(module.weight, a=0., b=1.)
        elif initialization_type == "normal":
            # drawn from normal distribution with mean 0 and standard deviation 1
            torch.nn.init.normal_(module.weight, mean=0., std=1.)
        elif initialization_type == "zeros":
            # initialize with all zeros
            torch.nn.init.zeros_(module.weight)
        elif initialization_type == "ones":
            # initialize with all ones
            torch.nn.init.ones_(module.weight)
        else:
            print(f"ERROR: Unsupported module weight initialization type {initialization_type}")

        # initialize bias with bias_init_value
        # default bias_init_value=0
        module.bias.data.fill_(bias_init_value)
    except AttributeError as ae:
        # ignore layers which do not have the weight and/or bias attributes
        print(f"WARNING: {ae}")
        pass
