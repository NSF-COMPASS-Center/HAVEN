import os
import pandas as pd
import random
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from utils import kmer_utils, utils
from prediction.models import logistic_regression


def execute(config):
    # input settings
    input_settings = config["input_settings"]
    input_dir = input_settings["input_dir"]
    input_dataset_dir = input_settings["dataset_dir"]
    input_files = input_settings["file_names"]
    input_files = [os.path.join(input_dir, input_dataset_dir, input_file) for input_file in input_files]

    # output settings
    output_settings = config["output_settings"]
    output_dir = output_settings["output_dir"]
    output_dataset_dir = output_settings["dataset_dir"]
    output_prefix = output_settings["prefix"]
    output_prefix = "_" + output_prefix if output_prefix is not None else ""

    # classification settings
    classification_settings = config["classification_settings"]
    k = classification_settings["kmer_settings"]["k"]
    n = classification_settings["n_iterations"]
    classification_type = classification_settings["type"]
    train_proportion = classification_settings["train_proportion"]
    models = classification_settings["models"]

    label_settings = classification_settings["label_settings"]
    label_col = label_settings["label_col"]

    df = read_dataset(input_files, label_col)
    kmer_df = kmer_utils.compute_kmer_based_dataset(df, k, label_col)
    transformed_df_with_label = transform_labels(kmer_df, classification_type, label_settings)

    # perform classification
    for model in models:
        if model["active"] is False:
            print(f"Skipping {model['name']} ...")
            continue
        model_name = model["name"]
        output_file_name = f"kmer_k{k}_{model_name}_{label_col}_{classification_type}_tr{train_proportion}_n{n}" + output_prefix + "_output.csv"
        output_file_path = os.path.join(output_dir, output_dataset_dir, output_file_name)

        # Setting needed values within model object for cleaner code and avoid passing multiple arguments.
        model["n"] = n
        model["train_proportion"] = train_proportion
        model["label_col"] = label_col
        model["classification_type"] = classification_type

        if model["name"] == "lr":
            print("Executing Logistic Regression")
            output_df = execute_lr_classification(transformed_df_with_label, model)
        elif model["name"] == "svm":
            print("Executing SVM")
            return

        print(f"Writing output of {model_name} to {output_file_path}")
        write_output(output_df, output_file_path)


def read_dataset(input_files, label):
    datasets = []
    for input_file in input_files:
        df = pd.read_csv(input_file, usecols=["id", "sequence", label])
        print(f"input file: {input_file}, size = {df.shape}")
        datasets.append(df)

    dataset = pd.concat(datasets)
    dataset.set_index("id", inplace=True)
    print(f"Size of input dataset = {dataset.shape}")
    # print(dataset)
    return dataset


def transform_labels(df, classification_type, label_settings):
    label_col = label_settings["label_col"]

    # remove rows with labels to be excluded
    df = df[~df[label_col].isin(label_settings["exclude_labels"])]
    labels = df[label_col].unique()
    if classification_type == "binary":
        positive_label = label_settings["positive_label"]
        negative_label = "Not " + positive_label
        df[label_col] = np.where(df[label_col] == positive_label, positive_label, negative_label)
        labels = [negative_label, positive_label]

    label_idx_map, idx_label_map = utils.get_label_vocabulary(labels)
    print(f"label_idx_map={label_idx_map}\nidx_label_map={idx_label_map}")

    df[label_col] = df[label_col].transform(lambda x: label_idx_map[x])
    print(df[label_col].unique())
    return df


def execute_lr_classification(df, model):
    results = []
    for i in range(model["n"]):
        print(f"Iteration {i}")
        X_train, X_test, y_train, y_test = create_splits(df, model["train_proportion"], model["label_col"])
        y_pred = logistic_regression.run(X_train, X_test, y_train, model)
        result_df = pd.DataFrame(y_pred)
        result_df["itr"] = i
        result_df["y_true"] = y_test

        print(f"result size = {result_df.shape}")
        results.append(result_df)
    return pd.concat(results, axis=1)


def create_splits(df, train_proportion, label_col):
    seed = random.randint(0, 10000)
    print(f"seed={seed}")
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=[label_col]).values, df[label_col].values,
                                                        train_size=train_proportion, random_state=seed)
    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.fit_transform(X_test)
    print(f"X_train size = {X_train.shape}")
    print(f"X_test size = {X_test.shape}")
    print(f"y_train size = {y_train.shape}")
    print(f"y_test size = {y_test.shape}")
    return X_train, X_test, y_train, y_test


def write_output(df, output_file_path):
    df.to_csv(output_file_path, index=False)
