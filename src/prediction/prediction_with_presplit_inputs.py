import os
import pandas as pd
import random

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from utils import kmer_utils, utils
from prediction.models import logistic_regression


def execute(config):
    # input settings
    input_settings = config["input_settings"]
    input_dir = input_settings["input_dir"]
    inputs = input_settings["file_names"]

    # output settings
    output_settings = config["output_settings"]
    output_dir = output_settings["output_dir"]
    output_dataset_dir = output_settings["dataset_dir"]
    output_prefix = output_settings["prefix"]
    output_prefix = "_" + output_prefix if output_prefix is not None else ""

    # classification settings
    classification_settings = config["classification_settings"]
    k = classification_settings["kmer_settings"]["k"]
    classification_type = classification_settings["type"]
    models = classification_settings["models"]

    label_settings = classification_settings["label_settings"]
    label_col = label_settings["label_col"]

    results = {}
    itr = 0
    for input in inputs:
        print(f"Iteration {itr}")
        # 1. Read the data files
        train_df, test_df = read_dataset(input_dir, input, label_col)
        df = pd.concat([train_df, test_df])

        # 2. filter out noise: labels configured to be excluded, NaN labels
        df = utils.filter_noise(df, label_settings)

        # 3. Compute kmer features
        kmer_df = kmer_utils.compute_kmer_features(df, k, label_col)
        # get the split column again to distinguish train and test datasets
        kmer_df = kmer_df.join(df["split"], on="id", how="left")

        # 4. Group the labels (if applicable) and convert the string labels to mapped integer indices
        kmer_df_with_transformed_label, idx_label_map = utils.transform_labels(kmer_df, classification_type, label_settings)

        # 5. Perform classification
        for model in models:
            if model["active"] is False:
                print(f"Skipping {model['name']} ...")
                continue
            model_name = model["name"]
            if model_name not in results:
                # first iteration
                results[model_name] = []

            # Set necessary values within model object for cleaner code and to avoid passing multiple arguments.
            model["label_col"] = label_col
            model["classification_type"] = classification_type

            if model["name"] == "lr":
                print("Executing Logistic Regression")
                result_df = execute_lr_classification(kmer_df_with_transformed_label, model, itr)
            else:
                continue

            # Remap the class indices to original input labels
            result_df.rename(columns=idx_label_map, inplace=True)
            result_df["y_true"] = result_df["y_true"].map(idx_label_map)
            result_df["itr"] = itr

            results[model_name].append(result_df)
            itr += 1

    for model_name, result_dfs in results.items():
        output_file_name = f"kmer_k{k}_{model_name}_{label_col}_{classification_type}_presplit" + output_prefix + "_output.csv"
        output_file_path = os.path.join(output_dir, output_dataset_dir, output_file_name)
        # create any missing parent directories
        Path(os.path.dirname(output_file_path)).mkdir(parents=True, exist_ok=True)
        # 5. Write the classification output
        print(f"Writing results of {model_name} to {output_file_path}")
        pd.concat(result_dfs).to_csv(output_file_path, index=False)


def read_dataset(input_dir, input, label_col):
    train_datasets = []
    test_datasets = []
    sub_dir = input["dir"]

    train_files = input["train"]
    for train_file in train_files:
        input_file_path = os.path.join(input_dir, sub_dir, train_file)
        df = pd.read_csv(input_file_path, usecols=["id", "sequence", label_col])
        print(f"input train file: {input_file_path}, size = {df.shape}")
        train_datasets.append(df)

    test_files = input["test"]
    for test_file in test_files:
        input_file_path = os.path.join(input_dir, sub_dir, test_file)
        df = pd.read_csv(input_file_path, usecols=["id", "sequence", label_col])
        print(f"input test file: {input_file_path}, size = {df.shape}")
        test_datasets.append(df)

    train_df = pd.concat(train_datasets)
    train_df["split"] = "train"
    train_df.set_index("id", inplace=True)
    print(f"Size of input train dataset = {train_df.shape}")

    test_df = pd.concat(test_datasets)
    test_df["split"] = "test"
    test_df.set_index("id", inplace=True)
    print(f"Size of input test dataset = {test_df.shape}")

    return train_df, test_df


def execute_lr_classification(df, model, itr):
    label_col = model["label_col"]
    drop_cols = ["split", label_col]

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[label_col]

    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df[label_col]

    # Standardize dataset
    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.fit_transform(X_test)
    # Perform classification
    y_pred = logistic_regression.run(X_train, X_test, y_train, model)

    result_df = pd.DataFrame(y_pred)
    result_df["y_true"] = y_test.values
    print(f"result size = {result_df.shape}")

    return result_df