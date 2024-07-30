import math
import random

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import pandas as pd
import os

from utils import utils, kmer_utils, constants
from datasets.collations.padding import Padding, PaddingUnlabeled
from datasets.collations.padding_with_id import PaddingWithID
from datasets.protein_sequence_dataset import ProteinSequenceDataset
from datasets.protein_sequence_with_label_dataset import ProteinSequenceWithLabelDataset
from datasets.protein_sequence_unlabeled_dataset import ProteinSequenceUnlabeledDataset
from datasets.protein_sequence_with_id_dataset import ProteinSequenceDatasetWithID
from datasets.protein_sequence_kmer_dataset import ProteinSequenceKmerDataset
from datasets.protein_sequence_cgr_dataset import ProteinSequenceCGRDataset
from datasets.collations.fsl_episode import FewShotLearningEpisode
from datasets.samplers.fsl_fixed_task_sampler import FewShotLearningFixedTaskSampler
from datasets.samplers.fsl_varying_task_sampler import FewShotLearningVaryingTaskSampler
from datasets.samplers.fsl_test_task_sampler import FewShotLearningTestTaskSampler


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


def split_dataset_for_few_shot_learning(df, label_col, train_proportion=0.7, val_proportion=0.1, test_proportion=0.2, seed=0):
    print(f"Splitting dataset based on '{label_col}' with seed={seed}, "
          f"train_proportion={train_proportion}, "
          f"val_proportion={val_proportion}, "
          f"and test_proportion={test_proportion}")
    labels = list(df[label_col].unique())
    n_labels = len(labels)
    n_train_labels = int(math.floor(n_labels * train_proportion))
    n_val_labels = int(math.floor(n_labels * val_proportion))
    n_test_labels = int(math.floor(n_labels * test_proportion))

    print(f"# unique labels = {n_labels}\n"
          f"# train labels = {n_train_labels}\n"
          f"# val labels = {n_val_labels}\n"
          f"# test labels = {n_test_labels}")
    random.seed(seed)
    train_labels = set(random.sample(labels, n_train_labels))

    # Random sampling from a set is deprecated since Python 3.9 and will be removed in a subsequent version.
    # Hack: Convert 'labels' back to list
    labels = list(set(labels) - train_labels)
    val_labels = set(random.sample(labels, n_val_labels))

    labels = list(set(labels) - val_labels)
    test_labels = random.sample(labels, n_test_labels)

    train_df = df[df[label_col].isin(list(train_labels))]
    val_df = df[df[label_col].isin(list(val_labels))]
    test_df = df[df[label_col].isin(test_labels)]

    print(f"Training: # samples = {train_df.shape[0]}, # labels = {len(train_labels)}")
    print(f"Validation: # samples = {val_df.shape[0]}, # labels = {len(val_labels)}")
    print(f"Testing: # samples = {test_df.shape[0]}, # labels = {len(test_labels)}")

    return train_df, val_df, test_df


def load_dataset(input_dir, input_file_names, sequence_settings, cols, label_settings, label_col, classification_type):
    df = read_dataset(input_dir, input_file_names, cols=cols)
    return load_dataset_with_df(df[cols], sequence_settings, label_settings, label_col, classification_type)


def load_dataset_with_df(df, sequence_settings, label_settings, label_col, classification_type):
    df, index_label_map = utils.transform_labels(df, label_settings, classification_type=classification_type)
    dataset_loader = get_dataset_loader(df, sequence_settings, label_col)
    return index_label_map, dataset_loader


def load_kmer_dataset(input_dir, input_file_names, seed, train_proportion,
                      id_col, seq_col, label_col, label_settings, classification_type, k, kmer_keys=None):
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
    truncate = sequence_settings["truncate"]
    split_sequence = sequence_settings["split_sequence"]

    dataset = None
    collate_func = None
    if exclude_label:
        dataset = ProteinSequenceUnlabeledDataset(df, seq_col, max_seq_len, truncate, split_sequence, sequence_settings["cls_token"])
        collate_func = PaddingUnlabeled(max_seq_len, sequence_settings["cls_token"])
    else:
        dataset = ProteinSequenceDataset(df, seq_col, label_col, truncate, max_seq_len)
        collate_func = Padding(max_seq_len)

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_func)


def get_token_with_id_dataset_loader(df, sequence_settings, label_col):
    seq_col = sequence_settings["sequence_col"]
    id_col = sequence_settings["id_col"]
    batch_size = sequence_settings["batch_size"]
    max_seq_len = sequence_settings["max_sequence_length"]
    truncate = sequence_settings["truncate"]


    dataset = ProteinSequenceDatasetWithID(df, id_col, seq_col, max_seq_len, truncate, label_col)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                      collate_fn=PaddingWithID(max_seq_len))


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


def get_few_shot_learning_task_sampler(dataset, few_shot_learn_settings):
    n_way_type = few_shot_learn_settings["n_way_type"]
    n_way = few_shot_learn_settings["n_way"] # incase of varying, n_way will contain a range as list(int)
    n_shot = few_shot_learn_settings["n_shot"]
    n_query = few_shot_learn_settings["n_query"]
    n_task = few_shot_learn_settings["n_task"]

    task_sampler = None
    if n_way_type == "varying":
        task_sampler = FewShotLearningVaryingTaskSampler(dataset=dataset,
                                                         n_way_range=n_way,
                                                         n_shot=n_shot,
                                                         n_query=n_query,
                                                         n_task=n_task)
    elif n_way_type == "fixed":
        # further subtyping based on number of query samples to determine training or testing dataset
        if n_query == -1:
            # test dataset_loader, use a different task sampler
            task_sampler = FewShotLearningTestTaskSampler(dataset=dataset,
                                                          n_way=n_way,
                                                          n_shot=n_shot,
                                                          n_task=n_task)
        else:
            task_sampler = FewShotLearningFixedTaskSampler(dataset=dataset,
                                                           n_way=n_way,
                                                           n_shot=n_shot,
                                                           n_query=n_query,
                                                           n_task=n_task)
    return task_sampler


def get_episodic_dataset_loader(df, sequence_settings, label_col, few_shot_learn_settings):
    dataset = ProteinSequenceWithLabelDataset(df=df,
                                     sequence_col=sequence_settings["sequence_col"],
                                     max_seq_len=sequence_settings["max_sequence_length"],
                                     truncate=sequence_settings["truncate"],
                                     label_col=label_col)

    fsl_episode = FewShotLearningEpisode(n_shot=few_shot_learn_settings["n_shot"],
                                         n_query=few_shot_learn_settings["n_query"],
                                         max_length=sequence_settings["max_sequence_length"])

    return DataLoader(dataset=dataset,
                      batch_sampler=get_few_shot_learning_task_sampler(dataset, few_shot_learn_settings),
                      collate_fn=fsl_episode)