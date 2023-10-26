import random

import joblib
import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import RandomOverSampler
from pathlib import Path
import os
from sklearn.utils.class_weight import compute_class_weight
import yaml

### functions related to labels, grouping, and label vocabulary

def filter_noise(df, label_settings):
    label_col = label_settings["label_col"]
    # remove rows with labels to be excluded
    df = df[~df[label_col].isin(label_settings["exclude_labels"])]
    # remove rows with Nan label values
    df = df[~df[label_col].isna()]
    return df


def transform_labels(df, label_settings, classification_type=None, silent=False):
    label_col = label_settings["label_col"]
    if "label_groupings" in label_settings.keys():
        label_grouping_config = label_settings["label_groupings"]
        if not silent:
            print(f"Grouping labels using config : {label_grouping_config}")
        df = group_labels(df, label_col, label_grouping_config)

    # labels = df[label_col].unique()
    labels = list(label_grouping_config.keys())

    if classification_type == "binary":
        positive_label = label_settings["positive_label"]
        negative_label = "Not " + positive_label
        df[label_col] = np.where(df[label_col] == positive_label, positive_label, negative_label)
        labels = [negative_label, positive_label]

    label_idx_map, idx_label_map = get_label_vocabulary(labels)
    if not silent:
        print(f"label_idx_map={label_idx_map}\nidx_label_map={idx_label_map}")

    df[label_col] = df[label_col].transform(lambda x: label_idx_map[x])
    return df, idx_label_map


def group_labels(df, label_col, label_grouping_config):
    group_others = False
    group_others_key = None
    for k, v in label_grouping_config.items():
        if len(v) == 1 and v[0] == "*":
            group_others = True
            group_others_key = k
            continue
        df.loc[df[label_col].isin(v), label_col] = k
    if group_others:
        values = list(label_grouping_config.keys())
        df.loc[~df[label_col].isin(values), label_col] = group_others_key
    return df


def get_label_vocabulary(labels):
    labels.sort()
    label_idx_map = {}
    idx_label_map = {}

    for idx, label in enumerate(labels):
        label_idx_map[label] = idx
        idx_label_map[idx] = label
    return label_idx_map, idx_label_map


### functions related to class distributions

def compute_class_distribution(df, label_col, format=False):
    labels_counts = df[label_col].value_counts()
    n = labels_counts.sum()
    labels_counts = labels_counts / n * 100
    labels_counts = labels_counts.to_dict()
    if format:
        labels_counts = {k: f"{k} ({v:.2f}%)" for k, v, in labels_counts.items()}
    return labels_counts


def get_class_weights(datasetloader):
    labels = datasetloader.dataset.get_labels()
    class_weights = compute_class_weight(class_weight="balanced",
                                classes=np.unique(labels),
                                y=labels)
    return torch.tensor(class_weights, dtype=torch.float)


### functions related to writing outputs
def write_output(model_dfs, output_dir, output_filename_prefix, output_type):
    for model_name, dfs in model_dfs.items():
        output_file_name = f"{output_filename_prefix}_{model_name}_{output_type}.csv"
        output_file_path = os.path.join(output_dir, output_file_name)
        # create any missing parent directories
        Path(os.path.dirname(output_file_path)).mkdir(parents=True, exist_ok=True)
        # 5. Write the classification output
        print(f"Writing {output_type} of {model_name} to {output_file_path}")
        pd.concat(dfs).to_csv(output_file_path, index=True)


def write_output_model(model, output_dir, output_filename_prefix, model_name):
    output_file_name = f"{output_filename_prefix}_{model_name}_model.joblib"
    output_file_path = os.path.join(output_dir, output_file_name)
    # create any missing parent directories
    Path(os.path.dirname(output_file_path)).mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_file_path)


def get_validation_scores(cv_model):
    k = 5
    params = cv_model["params"]
    validation_scores = {}
    for i, param in enumerate(params):
        param_key = ""
        for param_name, param_value in param.items():
            param_key += (param_name + "_" + str(param_value) + "_")
        param_scores = []
        for itr in range(k):
            # as per the key in the object returned by scikit
            param_score_key = "split" + str(itr) + "_test_score"
            param_scores.append(cv_model[param_score_key][i])
        validation_scores[param_key] = param_scores
    return pd.DataFrame(validation_scores)


def random_oversampling(X, y):
    vals, count = np.unique(y, return_counts=True)
    print(f"Label counts before resampling = {[*zip(vals, count)]}")
    random_oversampler = RandomOverSampler(random_state=random.randint(0, 1000))
    X_resampled, y_resampled = random_oversampler.fit_resample(X, y)
    vals, count = np.unique(y_resampled, return_counts=True)
    print(f"Label counts after resampling = {[*zip(vals, count)]}")
    return X_resampled, y_resampled


# Returns a config map for the yaml at the path specified
def parse_config(config_file_path):
    config = None
    try:
        with open(config_file_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
    except yaml.YAMLError as err:
        print(f"Error parsing config file: {err}")
    return config
