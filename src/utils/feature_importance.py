import os
import pandas as pd
from pathlib import Path
from utils import kmer_utils, utils, visualization_utils


def execute(config):
    input_settings = config["input_settings"]
    classification_settings = config["classification_settings"]
    id_col = classification_settings["id_col"]
    sequence_col = classification_settings["sequence_col"]
    label_col = input_settings["label_settings"]["label_col"]
    dataset_df = read_dataset(input_settings["dataset_dir"], input_settings["dataset_files"], input_settings["label_settings"])

    kmer_binary_df = get_kmer_binary_df(dataset_df, id_col, sequence_col, label_col)

    feature_imp_df = read_feature_imp_file(input_settings["feature_imp_dir"], input_settings["feature_imp_file"])
    feature_imp_settings = config["feature_importance_settings"]

    output_settings = config["output_settings"]

    if feature_imp_settings["prevalence"]:
        feature_prevalence_df = compute_prevalence(kmer_binary_df, label_col, output_settings)
    if feature_imp_settings["prevalence_by_host"]:
        compute_prevalence_by_host(kmer_binary_df, label_col, output_settings)
    if feature_imp_settings["feature_imp"]:
        compute_top_features(feature_imp_settings["feature_imp_k"], feature_imp_df, feature_prevalence_df, output_settings)
    if feature_imp_settings["feature_imp_by_prevalence"]:
        compute_feature_imp_prevalence_comparison(feature_imp_df, feature_prevalence_df, output_settings)


def read_dataset(dataset_dir, dataset_files, label_settings):
    datasets = []
    label_col = "host"
    for dataset_file in dataset_files:
        df = pd.read_csv(os.path.join(dataset_dir, dataset_file), usecols=["id", "sequence", label_col])
        print(f"input file: {dataset_file}, size = {df.shape}")
        datasets.append(df)

    dataset_df = pd.concat(datasets)
    dataset_df.set_index("id", inplace=True)
    print(f"Size of input dataset = {dataset_df.shape}")

    # 2. filter out noise: labels configured to be excluded, NaN labels
    dataset_df = utils.filter_noise(dataset_df, label_settings)
    print(f"dataset_df shape after filter = {dataset_df.shape}")
    return dataset_df


def read_feature_imp_file(feature_imp_dir, feature_imp_file):
    feature_imp_df = pd.read_csv(os.path.join(feature_imp_dir, feature_imp_file), index_col=0)
    feature_imp_df.drop(columns=["itr"], inplace=True)
    return feature_imp_df


def get_kmer_binary_df(dataset_df, id_col, sequence_col, label_col):
    # Compute kmer features
    k = 3
    kmer_df = kmer_utils.compute_kmer_features(dataset_df, k, id_col, sequence_col, label_col)
    print(f"kmer_df shape = {kmer_df.shape}")

    kmer_df_wo_label = kmer_df.drop(columns=[label_col])
    # remove label column to convert the kmer counts to binary i.e kmer present or absent
    kmer_df_binary_wo_label = kmer_df_wo_label.mask(kmer_df_wo_label > 0, 1)
    print(f"kmer_df_binary shape = {kmer_df_binary_wo_label.shape}")
    # rejoin the label colum
    kmer_df_binary_w_label = kmer_df_binary_wo_label.merge(kmer_df[label_col], how="left", on=id_col)
    return kmer_df_binary_w_label


def compute_prevalence(kmer_binary_df, label_col, output_settings):
    kmer_df_wo_label = kmer_binary_df.drop(columns=[label_col])
    n = kmer_df_wo_label.shape[0]
    print(f"n = {n}")

    # count number of occurrences of feature in df
    feature_count_df = kmer_df_wo_label.sum(axis=0)
    print(f"df_feature_count size = {feature_count_df.shape}")
    feature_prevalence_df = feature_count_df.transform(lambda x: x / n * 100)
    print(f"feature_prevalence_df size = {feature_prevalence_df.shape}")

    output_dir = output_settings["output_dir"]

    output_file_path = os.path.join(output_dir, output_settings["feature_imp_dir"], output_settings["dataset_dir"], output_settings["prefix"] + "_feature_prevalence.csv")
    # create any missing parent directories
    Path(os.path.dirname(output_file_path)).mkdir(parents=True, exist_ok=True)
    feature_prevalence_df.to_csv(output_file_path)

    output_file_path = os.path.join(output_dir, output_settings["visualization_dir"], output_settings["dataset_dir"], output_settings["prefix"] + "_feature_prevalence_distribution.png")
    # create any missing parent directories
    Path(os.path.dirname(output_file_path)).mkdir(parents=True, exist_ok=True)
    visualization_utils.feature_prevalence_distribution_plot(feature_prevalence_df, output_file_path)

    return feature_prevalence_df


def compute_prevalence_by_host(kmer_binary_df, label_col, output_settings):
    # count number of occurrences of feature by host type
    host_counts = kmer_binary_df[label_col].value_counts()
    feature_count_by_host_df = kmer_binary_df.groupby(label_col).sum()
    # instantiate a prevalence df with the counts and label column. These counts will then be converted to prevalence %
    feature_prevalence_by_host_df = feature_count_by_host_df.merge(host_counts, left_index=True, right_index=True)

    feature_prevalence_by_host_df_cols = list(feature_prevalence_by_host_df.columns)
    feature_prevalence_by_host_df_cols.remove(label_col)

    for col in feature_prevalence_by_host_df_cols:
        feature_prevalence_by_host_df[col] = feature_prevalence_by_host_df[col] / feature_prevalence_by_host_df["host"] * 100

    output_dir = output_settings["output_dir"]
    output_file_path = os.path.join(output_dir, output_settings["feature_imp_dir"], output_settings["dataset_dir"], output_settings["prefix"] + "_feature_prevalence_by_host.csv")
    # create any missing parent directories
    Path(os.path.dirname(output_file_path)).mkdir(parents=True, exist_ok=True)
    feature_prevalence_by_host_df.to_csv(output_file_path)


def compute_top_features(k, feature_imp_df, feature_prevalence_df, output_settings):
    abs_df = feature_imp_df.abs()
    mean_abs_df = abs_df.mean(axis=0)
    print(f"mean_abs_df size = {mean_abs_df.shape}")
    top_k_cols = mean_abs_df.nlargest(n=k).index
    print(f"top_k_cols = {top_k_cols}")

    top_k_df = feature_imp_df[top_k_cols]
    feature_prevalence = feature_prevalence_df.to_dict()
    top_k_df = pd.melt(top_k_df)
    top_k_df["variable"] = top_k_df["variable"].transform(
        lambda x: x + " (" + str(round(feature_prevalence[x], 2)) + "%)")
    output_file_path = os.path.join(output_settings["output_dir"], output_settings["visualization_dir"], output_settings["dataset_dir"], output_settings["prefix"] + "_top_" + str(k) + "_features.png")
    # create any missing parent directories
    Path(os.path.dirname(output_file_path)).mkdir(parents=True, exist_ok=True)
    visualization_utils.top_k_features_box_plot(top_k_df, output_file_path)


def compute_feature_imp_prevalence_comparison(feature_imp_df, feature_prevalence_df, output_settings):
    abs_df = feature_imp_df.abs()
    mean_abs_df = abs_df.mean(axis=0)
    print(f"mean_abs_df size = {mean_abs_df.shape}")
    df = pd.DataFrame({"imp": mean_abs_df, "prevalence": feature_prevalence_df})
    output_file_path = os.path.join(output_settings["output_dir"], output_settings["visualization_dir"],
                                    output_settings["dataset_dir"],
                                    output_settings["prefix"] + "_imp_by_prevalence.png")
    # create any missing parent directories
    Path(os.path.dirname(output_file_path)).mkdir(parents=True, exist_ok=True)
    visualization_utils.feature_imp_by_prevalence_scatter_plot(df, output_file_path)