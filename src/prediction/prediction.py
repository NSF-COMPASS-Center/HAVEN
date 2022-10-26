import os
import pandas as pd
import random
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from utils import kmer_utils, utils
from prediction.models import logistic_regression


def execute(config):
    input_settings = config["input_settings"]
    input_dir = input_settings["input_dir"]
    input_dataset_dir = input_settings["dataset_dir"]
    input_files = input_settings["file_names"]

    output_settings = config["output_settings"]
    output_dir = output_settings["output_dir"]
    output_dataset_dir = output_settings["dataset_dir"]
    output_prefix = output_settings["prefix"]
    output_prefix = "_" + output_prefix if output_prefix is not None else ""

    classification_settings = config["classification_settings"]
    k = classification_settings["kmer_settings"]["k"]
    n = classification_settings["n_iterations"]
    lr_settings = classification_settings["lr_settings"]
    label = lr_settings["label"]

    input_files = [os.path.join(input_dir, input_dataset_dir, input_file) for input_file in input_files]

    output_file_name = f"kmer_k{k}_lr_{label}_tr{lr_settings['train_proportion']}_n{n}" + output_prefix + "_output.csv"
    output_file_path = os.path.join(output_dir, output_dataset_dir, output_file_name)
    print(output_file_path)

    df = read_dataset(input_files, lr_settings["label"])
    transformed_df_with_label = kmer_utils.compute_kmer_based_dataset(df, k, label)

    label_idx_map, idx_label_map = utils.get_label_vocabulary(df[label].unique())
    print(f"label_idx_map={label_idx_map}\nidx_label_map={idx_label_map}")
    print(transformed_df_with_label)
    # transformed_df_with_label[label] = transformed_df_with_label[label].transform(lambda x: label_idx_map[x])
    transformed_df_with_label[label] = np.where(transformed_df_with_label[label] == "Human", 1, 0)
    print(transformed_df_with_label[label].unique())

    print("+++++++++++++++++++++++++++++++++++++++++++++++++\n")
    output_df = execute_classification(transformed_df_with_label, n, lr_settings)
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


def execute_classification(df, n, lr_settings):
    train_proportion = lr_settings["train_proportion"]
    label = lr_settings["label"]
    results = []
    for i in range(n):
        seed = random.randint(0, 10000)
        print(f"Iteration {i}: seed={seed}")
        X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=[label]).values, df[label].values, train_size=train_proportion, random_state=seed)
        min_max_scaler = MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)
        X_test = min_max_scaler.fit_transform(X_test)
        print(f"X_train size = {X_train.shape}")
        print(f"X_test size = {X_test.shape}")
        print(f"y_train size = {y_train.shape}")
        print(f"y_test size = {y_test.shape}")
        
        y_pred = logistic_regression.run(X_train, X_test, y_train, lr_settings)
        result_df = pd.DataFrame({"itr": i, "y_true": y_test, "y_pred": y_pred})
        print(f"result size = {result_df.shape}")
        results.append(result_df)
    return pd.concat(results, axis=1)


def write_output(df, output_file_path):
    df.to_csv(output_file_path, index=False)