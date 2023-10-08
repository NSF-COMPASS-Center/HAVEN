#!/usr/src/env python
import argparse
import yaml
import os
import pandas as pd
from pathlib import Path
import ast

NUCLEOTIDE = "nucleotide"
PROTEIN = "protein"
ID_COL = "id"
SEQ_COL = "seq"
NON_TOKEN = "-"

def parse_args():
    parser = argparse.ArgumentParser(description='Generate perturbed sequences for a given dataset')
    parser.add_argument("-if", "--input_files", required=True,
                        help="File(s) with input sequences.\n")
    parser.add_argument("-od", "--output_dir", required=True,
                        help="Absolute path to output directory.\n")
    parser.add_argument("-st", "--sequence_type", required=True,
                         help=f"Type of input sequences. Supported types: {NUCLEOTIDE}, {PROTEIN}")
    args = parser.parse_args()
    return args


def get_sequence_vocabulary(sequence_type):
    if sequence_type == NUCLEOTIDE:
        return ["A", "C", "G", "T"]
    elif sequence_type == PROTEIN:
        return ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    else:
        print(f"ERROR: Unsupported sequence type '{sequence_type}'. Supported values: {NUCLEOTIDE}, {PROTEIN}.")
        exit(1)

def perturb_sequence(id, sequence, sequence_vocab):
    df = pd.DataFrame(columns=[ID_COL, SEQ_COL])

    sequences = []

    # index: perturbed_position
    for index, orig_token in enumerate(sequence):
        # by replacing with every possible token in the vocab, we are creating a duplicate of the original sequence
        # we will remove these duplicate sequences at the end

        # create new sequence only for positions with aligned tokens
        if orig_token == NON_TOKEN:
            continue

        for new_token in sequence_vocab:
            # format for id = <orig_id>_<orig_token>_<index>_<new_token>
            new_id = f"{id}_{orig_token}_{index}_{new_token}"
            new_seq = sequence[:index] + new_token + sequence[index + 1:]
            sequences.append({ID_COL: new_id, SEQ_COL: new_seq})

    # create a dataframe
    df = pd.DataFrame(sequences)
    # drop all duplicate of the original sequences
    df.drop_duplicates(subset=SEQ_COL, inplace=True, ignore_index=True)
    return df


def generate_perturbed_sequences(input_files, output_dir, sequence_type):
    sequence_vocab = get_sequence_vocabulary(sequence_type)
    for input_file in input_files:
        input_file_name = os.path.basename(input_file)
        df = pd.read_csv(input_file)
        print(f"Read input file: {input_file}: {df.shape}")
        perturbed_sequences_count = 0
        for _, row in df.iterrows():
            id = row[ID_COL]
            perturbed_df = perturb_sequence(id, row[SEQ_COL], sequence_vocab)
            output_filepath = os.path.join(output_dir, f"{input_file_name}_{id}.csv")
            perturbed_df.to_csv(output_filepath, index=False)
            print(f"Processed {id}: {perturbed_df.shape} --> {output_filepath}")
            perturbed_sequences_count += perturbed_df.shape[0]

        print(f"Processed input file: {input_file}. Perturbed sequences count = {perturbed_sequences_count}")


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
    input_files = config.input_files.split(",")
    output_dir = config.output_dir
    sequence_type = config.sequence_type
    generate_perturbed_sequences(input_files, output_dir, sequence_type)
    return


if __name__ == '__main__':
    main()
    exit(0)
