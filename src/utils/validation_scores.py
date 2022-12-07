import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def execute(config):
    input_settings = config["input_settings"]
    input_dir = input_settings["val_scores_dir"]
    input_file = input_settings["val_scores_file"]
    val_scores_df = pd.read_csv(os.path.join(input_dir, input_file), index_col=0)

    output_settings = config["output_settings"]
    output_dir = output_settings["output_dir"]
    prefix = output_settings["prefix"]
    output_file_path = os.path.join(output_dir, prefix + "_val_scores.png")
    # create any missing parent directories
    Path(os.path.dirname(output_file_path)).mkdir(parents=True, exist_ok=True)
    plot_val_scores(val_scores_df, output_file_path)


def plot_val_scores(df, output_file_path):
    cols = list(df.columns)
    itrs = list(df["itr"].unique())
    cols.remove("itr")
    print(cols)
    df = df[df["itr"] == itrs[0]]
    df.drop(columns=["itr"], inplace=True)
    df["split"] = range(1,6)
    transformed_df = pd.melt(df, id_vars=["split"])
    print(transformed_df.head())
    plt.clf()
    sns.set_theme()
    sns.lineplot(data=transformed_df, x="variable", y="value", hue="split")
    # plt.rcParams['xtick.labelsize'] = 2
    plt.xticks(fontsize=8, rotation=90)
    plt.tight_layout()
    plt.savefig(output_file_path)
    # transformed_df = pd.melt(df, id_vars=["itr"], value_vars=cols)
    # print(transformed_df.head())
    # itrs = df["itr"].unique()
    # plt.clf()
    # sns.set_theme()
    # ax = None
    # for itr in itrs:
    #     transformed_df_itr = transformed_df[transformed_df["itr"] == itr]
    #     if ax is None:
    #         ax = sns.lineplot(data=transformed_df_itr, x="variable", y="value")
    #     else:
    #         sns.lineplot(data=transformed_df_itr, x="variable", y="value", ax= ax)
    #     plt.rcParams['xtick.labelsize'] = 8
    #     plt.tight_layout()
    #
    #     plt.xticks(rotation=20)
    # plt.savefig(output_file_path)
