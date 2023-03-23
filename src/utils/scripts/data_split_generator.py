#!/usr/src/env python
import argparse
import yaml
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description='Generate datasplits')
    parser.add_argument('-if', '--input_files', required=True,
                        help="File(s) to be split into training and testing sets.\n")
    parser.add_argument('-tp', '--train_proportion', required=True,
                        help="Fraction of dataset to be used for training.\n")
    parser.add_argument('-s', '--seed', required=True,
                        help="Seed to generate the split.\n")
    parser.add_argument('-od', '--output_dir', required=True,
                        help="Absolute path to output directory.\n")
    args = parser.parse_args()
    return args


# Returns a config map for the yaml at the path specified
def parse_config(config_file_path):
    config = None
    try:
        with open(config_file_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
    except yaml.YAMLError as err:
        print(f"Error parsing config file: {err}")
    return config


def generate_splits(input_files, train_proportion, seed, output_dir):
    for input_file in input_files:
        print(f"Input file = {input_file}")
        file_name = os.path.basename(input_file)
        train_dataset_file_path = os.path.join(output_dir, file_name + f"_tr{train_proportion}_train.csv")
        test_dataset_file_path = os.path.join(output_dir, file_name + f"_tr{train_proportion}_test.csv")
        # create any missing parent directories
        Path(os.path.dirname(train_dataset_file_path)).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(test_dataset_file_path)).mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(input_file)
        train_df, test_df = train_test_split(df, train_size=train_proportion, random_state=seed)
        train_df.to_csv(train_dataset_file_path, index=False)
        test_df.to_csv(test_dataset_file_path, index=False)
        print(f"Train file = {train_dataset_file_path}")
        print(f"Test file = {test_dataset_file_path}")


def main():
    config = parse_args()
    input_files = config.input_files.split(",")
    train_proportion = float(config.train_proportion)
    seed = int(config.seed)
    output_dir = config.output_dir
    generate_splits(input_files, train_proportion, seed, output_dir)
    return


if __name__ == '__main__':
    main()
