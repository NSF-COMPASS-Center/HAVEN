#!/usr/src/env python
import argparse
import yaml
import os
import pandas as pd
from pathlib import Path
import ast
import re

NUCLEOTIDE = "nucleotide"
PROTEIN = "protein"
ID_COL = "id"
SEQ_COL = "seq_aligned"
NON_TOKEN = "-"

def parse_args():
    parser = argparse.ArgumentParser(description='Generate perturbed sequences for a given dataset')
    parser.add_argument("-if", "--input_file", required=True,
                        help="File(s) with input sequences.\n")
    parser.add_argument("-id", "--input_dir", required=True,
                        help="Directory with all perturbed datasets.\n")
    parser.add_argument("-od", "--output_dir", required=True,
                        help="Absolute path to output directory.\n")
    args = parser.parse_args()
    return args


def process_files(input_file, input_dir, output_dir):
    sarscov2_human_df = pd.read_csv(input_file)
    sarscov2_human_uniprot_ids = sarscov2_human_df["id"].unique()
    print(f"Number of SARS-CoV-2 Human Sequences = {len(sarscov2_human_uniprot_ids)}")

    perturbed_input_files = os.listdir(input_dir)
    files_moved_count = 0
    for perturbed_input_file in perturbed_input_files:
         match = re.match(r".+\.csv_(\w+)\.csv", perturbed_input_file)
         perturbed_input_file_id = match.group(1)
         if perturbed_input_file_id in sarscov2_human_uniprot_ids:
                os.rename(os.path.join(input_dir, perturbed_input_file), os.path.join(output_dir, perturbed_input_file))
                files_moved_count += 1
    print(f"Moved {files_moved_count} files from {input_dir} to {output_dir}")

def main():
    config = parse_args()
    input_file = config.input_file
    input_dir = config.input_dir
    output_dir = config.output_dir
    process_files(input_file, input_dir, output_dir)
    return


if __name__ == '__main__':
    main()
    exit(0)
