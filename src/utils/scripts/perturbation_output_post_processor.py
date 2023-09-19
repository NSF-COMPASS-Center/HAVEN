#!/usr/src/env python
import argparse
import ast

import yaml
import os
import pandas as pd
from pathlib import Path

perturb_pos_col = "perturb_pos"
orig_token_col = "orig_token"
new_token_col = "new_token"
id_col = "id"

def parse_args():
    parser = argparse.ArgumentParser(description='Post process the output of perturbated dataset prediction')
    parser.add_argument("-id", "--input_dir", required=True,
                        help="Absolute path to input directory with all the prediction output files.\n")
    parser.add_argument("-od", "--output_dir", required=True,
                        help="Absolute path to output directory.\n")
    args = parser.parse_args()
    return args


def post_process_output(input_dir, output_dir):
    input_files = os.listdir(input_dir)
    for input_file in input_files:
        print(f"{input_file}")
        df = pd.read_csv(os.path.join(input_dir, input_file), converters={id_col: ast.literal_eval})
        df[id_col] = df[id_col].map(lambda x: x.pop())
        df[[id_col, orig_token_col, perturb_pos_col, new_token_col]] = df[id_col].str.split("_", expand=True)

        output_file_name = os.path.basename(input_file)
        # create any missing parent directories
        output_file_path = os.path.join(output_dir, output_file_name)
        Path(os.path.dirname(output_file_path)).mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file_path, index=False)


# Returns a config map for the yaml at the path specified
def parse_config(config_file_path):
    config = None
    try:
        with open(config_file_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
    except yaml.YAMLError as err:
        print(f"Error parsing config file: {err}")
    return config


def main():
    config = parse_args()
    input_dir = config.input_dir
    output_dir = config.output_dir
    post_process_output(input_dir, output_dir)
    return


if __name__ == '__main__':
    main()
    exit(0)
