import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


def execute(config):
    input_settings = config["input_settings"]
    input_dir = input_settings["input_dir"]
    input_dataset_dir = input_settings["dataset_dir"]
    input_files = input_settings["file_names"]

    output_settings = config["output_settings"]
    output_dir = output_settings["output_dir"]
    output_evaluation_dir = output_settings["evaluation_dir"]
    output_visualization_dir = output_settings["visualization_dir"]
    output_dataset_dir = output_settings["dataset_dir"]
    output_prefix = output_settings["prefix"]
    output_prefix = "_" + output_prefix if output_prefix is not None else ""
    input_file = input_files[0]
    # for input_file in input_files:

    input_file_path = os.path.join(input_dir, input_dataset_dir, input_file)
    output_file_name = Path(input_file_path).stem
    df = pd.read_csv(input_file)
    print(f"input results size = {df.shape}")
    print(df)

    evaluation_settings = config["evaluation_settings"]
    if evaluation_settings["auroc"]:
        output_file_path = os.path(output_dir, output_dataset_dir, output_visualization_dir, output_file_name + output_prefix + "_auroc.png")
        auroc(df, output_file_path)
    if evaluation_settings["auprc"]:
        output_file_path = os.path(output_dir, output_dataset_dir, output_visualization_dir,
                                   output_file_name + output_prefix + "_auprc.png")
        auprc(df, output_file_path)
    return


def auroc(df, output_file_path):
    itr_col = "itr"
    plt.clf()
    itrs = df[itr_col].unique()
    ax = plt.subplot(1, 1, 1)
    for itr in itrs:
        itr_df = df[df[itr_col] == itr]
        RocCurveDisplay.from_predictions(itr_df["y_test"].values, itr_df["y_pred"].values,
                                         name="Iteration: " + str(itr), ax=ax)
    plt.savefig(output_file_path)


def auprc(df, output_file_path):
    itr_col = "itr"
    plt.clf()
    itrs = df[itr_col].unique()
    ax = plt.subplot(1, 1, 1)
    for itr in itrs:
        itr_df = df[df[itr_col] == itr]
        PrecisionRecallDisplay.from_predictions(itr_df["y_test"].values, itr_df["y_pred"].values,
                                                name="Iteration: " + str(itr), ax=ax)
    plt.savefig(output_file_path)