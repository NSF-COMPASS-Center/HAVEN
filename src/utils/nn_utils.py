from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import copy

from prediction.datasets.protein_sequence_dataset import ProteinSequenceDataset
from prediction.datasets.protein_sequence_with_id_dataset import ProteinSequenceDatasetWithID
from prediction.datasets.protein_sequence_kmer_dataset import ProteinSequenceKmerDataset
from prediction.datasets.protein_sequence_cgr_dataset import ProteinSequenceCGRDataset
from utils.nlp_utils.padding import Padding
from utils.nlp_utils.padding_with_id import PaddingWithID

from utils.focal_loss import FocalLoss


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


def get_dataset_loader(df, sequence_settings, label_col, include_id_col=False):
    feature_type = sequence_settings["feature_type"]
    # supported values: kmer, cgr, token
    if feature_type == "kmer":
        return get_kmer_dataset_loader(df, sequence_settings, label_col)
    elif feature_type == "cgr":
        return get_cgr_dataset_loader(df, sequence_settings, label_col)
    elif feature_type == "token":
        if include_id_col:
            return get_token_with_id_dataset_loader(df, sequence_settings, label_col)
        else:
            return get_token_dataset_loader(df, sequence_settings, label_col)
    else:
        print(f"ERROR: Unsupported feature type: {feature_type}")


def get_token_dataset_loader(df, sequence_settings, label_col):
    seq_col = sequence_settings["sequence_col"]
    batch_size = sequence_settings["batch_size"]
    max_seq_len = sequence_settings["max_sequence_length"]
    pad_sequence_val = sequence_settings["pad_sequence_val"]
    truncate = sequence_settings["truncate"]

    dataset = ProteinSequenceDataset(df, seq_col, max_seq_len, truncate, label_col)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                      collate_fn=Padding(max_seq_len, pad_sequence_val))


def get_token_with_id_dataset_loader(df, sequence_settings, label_col):
    seq_col = sequence_settings["sequence_col"]
    id_col = sequence_settings["id_col"]
    batch_size = sequence_settings["batch_size"]
    max_seq_len = sequence_settings["max_sequence_length"]
    pad_sequence_val = sequence_settings["pad_sequence_val"]
    truncate = sequence_settings["truncate"]


    dataset = ProteinSequenceDatasetWithID(df, id_col, seq_col, max_seq_len, truncate, label_col)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                      collate_fn=PaddingWithID(max_seq_len, pad_sequence_val))



def get_kmer_dataset_loader(df, sequence_settings, label_col):
    dataset = ProteinSequenceKmerDataset(df,
                                         id_col=sequence_settings["id_col"],
                                         sequence_col=sequence_settings["sequence_col"],
                                         label_col=label_col,
                                         k=sequence_settings["kmer_settings"]["k"],
                                         kmer_keys=sequence_settings["kmer_keys"])
    return DataLoader(dataset=dataset, batch_size=sequence_settings["batch_size"], shuffle=True)


def get_cgr_dataset_loader(df, sequence_settings, label_col):
    dataset = ProteinSequenceCGRDataset(df,
                                        id_col=sequence_settings["id_col"],
                                        label_col=label_col,
                                        img_dir=sequence_settings["cgr_settings"]["img_dir"],
                                        img_size=sequence_settings["cgr_settings"]["img_size"])
    return DataLoader(dataset=dataset, batch_size=sequence_settings["batch_size"], shuffle=True)


def get_criterion(loss, class_weights=None):
    criterion = nn.CrossEntropyLoss()  # default
    if loss == "MultiMarginLoss":
        criterion = nn.MultiMarginLoss()
    if loss == "FocalLoss":
        criterion = FocalLoss(alpha=class_weights, gamma=2)
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
