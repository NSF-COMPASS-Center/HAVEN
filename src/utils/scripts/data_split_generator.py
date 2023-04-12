#!/usr/src/env python
import argparse
import yaml
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import ast


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
    parser.add_argument('-os', '--output_suffix', required=True,
                        help="Suffix to be added to the file names.\n")
    parser.add_argument('-train', '--train', required=True,
                        help="Boolean. Generate training files\n")
    parser.add_argument('-test', '--test', required=True,
                        help="Boolean. Generate testing files\n")
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


def generate_splits(input_files, train_proportion, seed, output_dir, output_suffix="", generate_train=True, generate_test=True):
    for input_file in input_files:
        print(f"Input file = {input_file}")
        file_name = os.path.basename(input_file)
        train_dataset_file_path = os.path.join(output_dir, file_name + f"_tr{train_proportion}_train{output_suffix}.csv")
        test_dataset_file_path = os.path.join(output_dir, file_name + f"_tr{train_proportion}_test{output_suffix}.csv")
        # create any missing parent directories
        Path(os.path.dirname(train_dataset_file_path)).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(test_dataset_file_path)).mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(input_file)
        train_df, test_df = train_test_split(df, train_size=train_proportion, random_state=seed)
        if generate_train:
            train_df.to_csv(train_dataset_file_path, index=False)
            print(f"Train file = {train_dataset_file_path}")
        if generate_test:
            test_df.to_csv(test_dataset_file_path, index=False)
            print(f"Test file = {test_dataset_file_path}")


def main():
    config = parse_args()
    input_files = config.input_files.split(",")
    train_proportion = float(config.train_proportion)
    seed = int(config.seed)
    output_dir = config.output_dir
    output_suffix = config.output_suffix
    generate_train = ast.literal_eval(config.train)
    generate_test = ast.literal_eval(config.test)
    generate_splits(input_files, train_proportion, seed, output_dir, output_suffix, generate_train, generate_test)
    return


if __name__ == '__main__':
    main()
