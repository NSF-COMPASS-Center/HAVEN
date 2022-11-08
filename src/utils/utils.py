import numpy as np


def filter_noise(df, label_settings):
    label_col = label_settings["label_col"]
    # remove rows with labels to be excluded
    df = df[~df[label_col].isin(label_settings["exclude_labels"])]
    # remove rows with Nan label values
    df = df[~df[label_col].isna()]
    return df


def transform_labels(df, classification_type, label_settings):
    label_col = label_settings["label_col"]
    if label_settings["label_groupings"] is not None:
        label_grouping_config = label_settings["label_groupings"]
        print(f"Grouping labels using config : {label_grouping_config}")
        df = group_labels(df, label_col, label_grouping_config)

    labels = df[label_col].unique()

    if classification_type == "binary":
        positive_label = label_settings["positive_label"]
        negative_label = "Not " + positive_label
        df[label_col] = np.where(df[label_col] == positive_label, positive_label, negative_label)
        labels = [negative_label, positive_label]

    label_idx_map, idx_label_map = get_label_vocabulary(labels)
    print(f"label_idx_map={label_idx_map}\nidx_label_map={idx_label_map}")

    df[label_col] = df[label_col].transform(lambda x: label_idx_map[x])
    print(df[label_col].unique())
    return df, idx_label_map


def group_labels(df, label_col, label_grouping_config):
    for k, v in label_grouping_config.items():
        df.loc[df[label_col].isin(v), label_col] = k
    return df


def get_label_vocabulary(labels):
    print(labels)
    label_idx_map = {}
    idx_label_map = {}

    for idx, label in enumerate(labels):
        label_idx_map[label] = idx
        idx_label_map[idx] = label
    return label_idx_map, idx_label_map
