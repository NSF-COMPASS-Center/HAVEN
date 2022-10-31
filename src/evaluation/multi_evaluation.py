import os
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import visualization_utils
import numpy as np


def execute(evaluation_settings, df, evaluation_output_file_base_path, visualization_output_file_base_path, output_file_name):
    evaluation_output_file_path = os.path.join(evaluation_output_file_base_path, output_file_name)
    visualization_output_file_path = os.path.join(visualization_output_file_base_path, output_file_name)
    itr_col = "itr"
    y_true_col = "y_true"
    y_pred_columns = list(df.columns.values)
    print(y_pred_columns)
    y_pred_columns.remove(itr_col)
    y_pred_columns.remove(y_true_col)
    print(f"y_pred_columns = {y_pred_columns}")
    if evaluation_settings["auroc"]:
        auroc(df, y_pred_columns, evaluation_output_file_path, visualization_output_file_path)
    if evaluation_settings["auprc"]:
        auprc(df, y_pred_columns, evaluation_output_file_path, visualization_output_file_path)
    return


def auroc(df, y_pred_columns, evaluation_output_file_path, visualization_output_file_path):
    # TODO: Move hard-coded values across src to common location (properties file?)
    itr_col = "itr"
    y_true_col = "y_true"
    itrs = df["itr"].unique()
    result = []
    for itr in itrs:
        df_itr = df[df[itr_col] == itr]
        print(f"y_true={df_itr[y_true_col]}")
        print(f"y_score={df_itr[y_pred_columns]}")

        y_true_df = convert_multiclass_label_to_binary(df_itr[y_true_col], y_pred_columns)
        print(f"y_true_df = {y_true_df}")
        print("sum")
        print(y_true_df.sum(axis=0))
        auroc_itr = roc_auc_score(y_true=y_true_df, y_score=df_itr[y_pred_columns], multi_class="ovr")
        result.append({itr_col: itr, "auroc": auroc_itr})
    result_df = pd.DataFrame(result)
    result_df.to_csv(evaluation_output_file_path + "_auroc.csv")
    visualization_utils.box_plot(result_df, "auroc", visualization_output_file_path + "_auroc_boxplot.png")
    return


def auprc(df, y_pred_columns, evaluation_output_file_path, visualization_output_file_path):
    # TODO: Move hard-coded values across src to common location (properties file?)
    itr_col = "itr"
    y_true_col = "y_true"
    itrs = df["itr"].unique()
    result = []
    for itr in itrs:
        df_itr = df[df[itr_col] == itr]
        print(f"Number of unique classes = {df_itr[y_true_col].unique()}")
        y_true_df = convert_multiclass_label_to_binary(df_itr[y_true_col], y_pred_columns)
        print(f"y_true_df = {y_true_df}")
        auprc_itr = average_precision_score(y_true=y_true_df, y_score=df_itr[y_pred_columns].values)
        result.append({itr_col: itr, "auprc": auprc_itr})
    result_df = pd.DataFrame(result)
    result_df.to_csv(evaluation_output_file_path + "_auprc.csv")
    visualization_utils.box_plot(result_df, "auprc", visualization_output_file_path + "_auprc_boxplot.png")
    return


def convert_multiclass_label_to_binary(y, labels):
    print(f"Number of unique classes = {y.unique()}")
    n = len(y)
    y_bin = np.zeros((n, len(labels)))
    for idx, val in enumerate(y):
        print(val)
        y_bin[idx][int(val)] = 1
    return pd.DataFrame(y_bin, columns=labels)