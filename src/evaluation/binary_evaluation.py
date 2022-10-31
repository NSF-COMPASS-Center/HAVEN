import os
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import visualization_utils


def execute(evaluation_settings, df, evaluation_output_file_base_path, visualization_output_file_base_path, output_file_name):
    evaluation_output_file_path = os.path.join(evaluation_output_file_base_path, output_file_name)
    visualization_output_file_path = os.path.join(visualization_output_file_base_path, output_file_name)
    if evaluation_settings["auroc"]:
        auroc(df, evaluation_output_file_path, visualization_output_file_path)
    if evaluation_settings["auprc"]:
        auprc(df, evaluation_output_file_path, visualization_output_file_path)
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
    result_df.to_csv(evaluation_output_file_path + "_auroc.csv")
    visualization_utils.box_plot(result_df, "auroc", visualization_output_file_path + "_auroc_boxplot.png")
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
    result_df.to_csv(evaluation_output_file_path + "_auprc.csv")
    visualization_utils.box_plot(result_df, "auprc", visualization_output_file_path + "_auprc_boxplot.png")
    return