import os
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from utils import visualization_utils
import numpy as np


def execute(evaluation_settings, df, evaluation_output_file_base_path, visualization_output_file_base_path, output_file_name):
    evaluation_output_file_path = os.path.join(evaluation_output_file_base_path, output_file_name)
    visualization_output_file_path = os.path.join(visualization_output_file_base_path, output_file_name)
    if evaluation_settings["auroc"]:
        auroc(df, evaluation_output_file_path, visualization_output_file_path)
    if evaluation_settings["auprc"]:
        auprc(df, evaluation_output_file_path, visualization_output_file_path)
    if evaluation_settings["accuracy"]:
        accuracy(df, evaluation_output_file_path, visualization_output_file_path)
    if evaluation_settings["f1"]:
        f1(df, evaluation_output_file_path, visualization_output_file_path)
    plot_prediction_distribution(df, evaluation_output_file_path, visualization_output_file_path)
    return


def plot_prediction_distribution(df, evaluation_output_file_path, visualization_output_file_path):
    itr_col = "itr"
    y_true_col = "y_true"
    itrs = df["itr"].unique()
    result = []

    for itr in itrs:
        df_itr = df[df[itr_col] == itr]
        y_pred = convert_probability_to_prediction(df_itr["1"].values, threshold=0.5)
        labels, counts = np.unique(y_pred, return_counts=True)
        result_itr_pred = pd.DataFrame({"label": labels, "label_count": counts, "itr": itr, "group": "y_pred"})

        y_true = df_itr[y_true_col].values
        labels, counts = np.unique(y_true, return_counts=True)
        result_itr_true = pd.DataFrame({"label": labels, "label_count": counts, "itr": itr, "group": "y_true"})

        result.append(result_itr_pred)
        result.append(result_itr_true)
    result_df = pd.concat(result)
    result_df.to_csv(evaluation_output_file_path + "class_distribution.csv")
    visualization_utils.class_distribution_plot(result_df, visualization_output_file_path + "class_distribution.png")


def accuracy(df, evaluation_output_file_path, visualization_output_file_path):
    # TODO: Move hard-coded values across src to common location (properties file?)
    itr_col = "itr"
    y_true_col = "y_true"
    itrs = df["itr"].unique()
    result = []

    for itr in itrs:
        df_itr = df[df[itr_col] == itr]
        y_pred = convert_probability_to_prediction(df_itr["1"].values, threshold=0.5)
        acc_itr = accuracy_score(y_true=df_itr[y_true_col].values, y_pred=y_pred)
        result.append({itr_col: itr, "accuracy": acc_itr})
    result_df = pd.DataFrame(result)
    result_df.to_csv(evaluation_output_file_path + "accuracy.csv")
    visualization_utils.box_plot(result_df, "accuracy", visualization_output_file_path + "accuracy_boxplot.png")
    return


def f1(df, evaluation_output_file_path, visualization_output_file_path):
    # TODO: Move hard-coded values across src to common location (properties file?)
    itr_col = "itr"
    y_true_col = "y_true"
    itrs = df["itr"].unique()
    result = []

    for itr in itrs:
        df_itr = df[df[itr_col] == itr]
        y_pred = convert_probability_to_prediction(df_itr["1"].values, threshold=0.5)
        f1_itr = f1_score(y_true=df_itr[y_true_col].values, y_pred=y_pred)
        result.append({itr_col: itr, "f1": f1_itr})
    result_df = pd.DataFrame(result)
    result_df.to_csv(evaluation_output_file_path + "f1.csv")
    visualization_utils.box_plot(result_df, "f1", visualization_output_file_path + "f1_boxplot.png")
    return


def auroc(df, evaluation_output_file_path, visualization_output_file_path):
    # TODO: Move hard-coded values across src to common location (properties file?)
    itr_col = "itr"
    y_true_col = "y_true"
    itrs = df["itr"].unique()
    result = []
    for itr in itrs:
        df_itr = df[df[itr_col] == itr]
        auroc_itr = roc_auc_score(y_true=df_itr[y_true_col].values, y_score=df_itr["1"].values, average="macro")
        result.append({itr_col: itr, "auroc": auroc_itr})
    result_df = pd.DataFrame(result)
    result_df.to_csv(evaluation_output_file_path + "auroc.csv")
    visualization_utils.box_plot(result_df, "auroc", visualization_output_file_path + "auroc_boxplot.png")
    return


def auprc(df, evaluation_output_file_path, visualization_output_file_path):
    # TODO: Move hard-coded values across src to common location (properties file?)
    itr_col = "itr"
    y_true_col = "y_true"
    itrs = df["itr"].unique()
    result = []
    for itr in itrs:
        df_itr = df[df[itr_col] == itr]
        auprc_itr = average_precision_score(y_true=df_itr[y_true_col].values, y_score=df_itr["1"].values, average="macro")
        result.append({itr_col: itr, "auprc": auprc_itr})
    result_df = pd.DataFrame(result)
    result_df.to_csv(evaluation_output_file_path + "auprc.csv")
    visualization_utils.box_plot(result_df, "auprc", visualization_output_file_path + "auprc_boxplot.png")
    return


def convert_probability_to_prediction(y_pred_prob, threshold):
    y_pred = [1 if y >= threshold else 0 for y in y_pred_prob]
    return y_pred