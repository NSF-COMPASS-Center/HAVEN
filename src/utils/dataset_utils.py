from sklearn.model_selection import train_test_split
import pandas as pd
import os
from utils import utils, nn_utils, kmer_utils

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


def split_dataset(df, seed, train_proportion, stratify_col=None):
    print(f"Splitting dataset with seed={seed}, train_proportion={train_proportion}, stratify_col={stratify_col}")
    train_df, test_df = train_test_split(df, train_size=train_proportion, random_state=seed, stratify=df[stratify_col])
    print(f"Size of train_dataset = {train_df.shape}")
    print(f"Size of test_dataset = {test_df.shape}")
    return train_df, test_df


def load_dataset(input_dir, input_file_names, sequence_settings, cols, label_settings, label_col, classification_type):
    df = read_dataset(input_dir, input_file_names, cols=cols)
    return load_dataset_with_df(df[cols], sequence_settings, cols, label_settings, label_col, classification_type)


def load_dataset_with_df(df, sequence_settings, label_settings, label_col):
    df, index_label_map = utils.transform_labels(df, label_settings, classification_type=classification_type)
    dataset_loader = nn_utils.get_dataset_loader(df, sequence_settings, label_col)
    return index_label_map, dataset_loader


def load_dataset(input_dir, input_file_names, seed, train_proportion, id_col, seq_col, label_col, label_settings, classification_type, k, kmer_keys=None):
    split_col = "split"
    df = read_dataset(input_dir, input_file_names,
                            cols=[id_col, seq_col, label_col])
    df, index_label_map = utils.transform_labels(df, label_settings, classification_type=classification_type)
    train_df, test_df = split_dataset(df, seed, train_proportion,
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