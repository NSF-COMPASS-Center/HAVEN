from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import pandas as pd
import os

from utils import utils, nn_utils, kmer_utils
from models.nlp.embedding.padding import Padding, PaddingUnlabeled
from datasets.protein_sequence_dataset import ProteinSequenceDataset
from datasets.protein_sequence_unlabeled_dataset import ProteinSequenceUnlabeledDataset
from datasets.protein_sequence_with_id_dataset import ProteinSequenceDatasetWithID
from datasets.protein_sequence_kmer_dataset import ProteinSequenceKmerDataset
from datasets.protein_sequence_cgr_dataset import ProteinSequenceCGRDataset


# read datasets using config properties
def read_dataset(input_dir, input_file_names, cols):
    datasets = []
    for input_file_name in input_file_names:
        input_file_path = os.path.join(input_dir, input_file_name)
        df = pd.read_csv(input_file_path, usecols=cols)
        print(f"input file: {input_file_path}, size = {df.shape}")
        datasets.append(df)

    df = pd.concat(datasets)
    print(f"Size of input dataset = {df.shape}")
    return df

def split_dataset(df, seed, train_proportion):
    print(f"Splitting dataset with seed={seed}, train_proportion={train_proportion}")
    train_df, test_df = train_test_split(df, train_size=train_proportion, random_state=seed)
    print(f"Size of train_dataset = {train_df.shape}")
    print(f"Size of test_dataset = {test_df.shape}")
    return train_df, test_df

def split_dataset_stratified(df, seed, train_proportion, stratify_col=None):
    print(f"Splitting dataset with seed={seed}, train_proportion={train_proportion}, stratify_col={stratify_col}")
    train_df, test_df = train_test_split(df, train_size=train_proportion, random_state=seed, stratify=df[stratify_col])
    print(f"Size of train_dataset = {train_df.shape}")
    print(f"Size of test_dataset = {test_df.shape}")
    return train_df, test_df


def load_dataset(input_dir, input_file_names, sequence_settings, cols, label_settings, label_col, classification_type):
    df = read_dataset(input_dir, input_file_names, cols=cols)
    return load_dataset_with_df(df[cols], sequence_settings, label_settings, label_col, classification_type)


def load_dataset_with_df(df, sequence_settings, label_settings, label_col, classification_type):
    df, index_label_map = utils.transform_labels(df, label_settings, classification_type=classification_type)
    dataset_loader = get_dataset_loader(df, sequence_settings, label_col)
    return index_label_map, dataset_loader


def load_kmer_dataset(input_dir, input_file_names, seed, train_proportion, id_col, seq_col, label_col, label_settings, classification_type, k, kmer_keys=None):
    split_col = "split"
    df = read_dataset(input_dir, input_file_names,
                            cols=[id_col, seq_col, label_col])
    df, index_label_map = utils.transform_labels(df, label_settings, classification_type=classification_type)
    train_df, test_df = split_dataset_stratified(df, seed, train_proportion,
                                                 stratify_col=label_col)
    train_df[split_col] = "train"
    test_df[split_col] = "test"
    df = pd.concat([train_df, test_df])
    print(f"Loaded dataset size = {df.shape}")

    kmer_df = kmer_utils.compute_kmer_features(df, k, id_col, seq_col, label_col, kmer_keys)
    print(f"kmer_df size = {kmer_df.shape}")

    kmer_df = kmer_df.join(df["split"], on=id_col, how="left")
    print(f"kmer_df size after join with split on id = {kmer_df.shape}")
    return index_label_map, kmer_df

def get_dataset_loader(df, sequence_settings, label_col=None, include_id_col=False, exclude_label=False):
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
            return get_token_dataset_loader(df, sequence_settings, label_col, exclude_label)
    else:
        print(f"ERROR: Unsupported feature type: {feature_type}")


def get_token_dataset_loader(df, sequence_settings, label_col, exclude_label):
    seq_col = sequence_settings["sequence_col"]
    batch_size = sequence_settings["batch_size"]
    max_seq_len = sequence_settings["max_sequence_length"]
    pad_sequence_val = sequence_settings["pad_token_val"]
    truncate = sequence_settings["truncate"]

    dataset = None
    collate_func = None
    if exclude_label:
        dataset = ProteinSequenceUnlabeledDataset(df, seq_col, max_seq_len, truncate)
        collate_func = PaddingUnlabeled(max_seq_len, pad_sequence_val)
    else:
        dataset = ProteinSequenceDataset(df, seq_col, max_seq_len, truncate, label_col)
        collate_func = Padding(max_seq_len, pad_sequence_val)

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_func)


def get_token_with_id_dataset_loader(df, sequence_settings, label_col):
    seq_col = sequence_settings["sequence_col"]
    id_col = sequence_settings["id_col"]
    batch_size = sequence_settings["batch_size"]
    max_seq_len = sequence_settings["max_sequence_length"]
    pad_sequence_val = sequence_settings["pad_token_val"]
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