import os
import pandas as pd
from pathlib import Path
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch
import tqdm
from statistics import mean
import wandb

from utils import utils, dataset_utils, nn_utils


def execute(config):
    # input settings
    input_settings = config["input_settings"]
    input_dir = input_settings["input_dir"]
    input_file_names = input_settings["file_names"]

    # output settings
    output_settings = config["output_settings"]
    output_dir = output_settings["output_dir"]
    results_dir = output_settings["results_dir"]
    sub_dir = output_settings["sub_dir"]
    output_prefix = output_settings["prefix"]
    output_prefix = output_prefix if output_prefix is not None else ""

    sequence_settings = config["sequence_settings"]
    label_settings = config["label_settings"]

    few_shot_learn_settings = config["few_shot_learn_settings"]
    meta_train_settings  = few_shot_learn_settings["meta_train_settings"]
    meta_validate_settings = few_shot_learn_settings["meta_validate_settings"]
    meta_test_settings = few_shot_learn_settings["meta_test_settings"]
    n_iters = few_shot_learn_settings["n_iterations"]

    id_col = sequence_settings["id_col"]
    sequence_col = sequence_settings["sequence_col"]
    label_col = label_settings["label_col"]

    wandb_config = {
        "n_epochs": few_shot_learn_settings["n_epochs"],
        "lr": few_shot_learn_settings["max_lr"],
        "max_sequence_length": sequence_settings["max_sequence_length"],
        "dataset": input_file_names[0]
    }

    results = {}
    for iter in range(n_iters):
        print(f"Iteration {iter}")
        # 1. Read the data files
        df = dataset_utils.read_dataset(input_dir, input_file_names,
                                        cols=[id_col, sequence_col, label_col])

        # 2. Transform labels
        df, index_label_map = utils.transform_labels(df, label_settings,
                                                     classification_type=classification_settings["type"])

        train_dataset_loader = None
        val_dataset_loader = None
        test_dataset_loader = None

        # 3. Split dataset
        if classification_settings["split_input"]:
            input_split_seeds = input_settings["split_seeds"]
            train_df, val_df, test_df = dataset_utils.split_dataset_for_few_shot_learning(df, label_col=label_col,
                                                                                          train_proportion=few_shot_learn_settings["train_proportion"],
                                                                                          val_proportion=few_shot_learn_settings["val_proportion"],
                                                                                          test_proportion=few_shot_learn_settings["test_proportion"],
                                                                                          seed=input_split_seeds[iter])
        pre_trained_models = config["pre_trained_models"]

