#!/usr/src/env python
import argparse
import yaml
import os
import pandas as pd
import math
from src.data_processing.cgr_fcgr.cgr import CGR


def parse_args():
    parser = argparse.ArgumentParser(description="Generate CGR images for sequences")
    parser.add_argument("-if", "--input_file", required=True,
                        help="File with sequences.\n")
    parser.add_argument("-id_col", "--id_col", required=True,
                        help="Name of the column containing the id\n")
    parser.add_argument("-seq_col", "--seq_col", required=True,
                        help="Name of the column containing the sequence\n")
    parser.add_argument("-st", "--sequence_type", required=True,
                        help="Type of sequence. Supported values: 'nucleotide', 'aminoacid'\n")
    parser.add_argument("-od", "--output_dir", required=True,
                        help="Absolute path to output directory.\n")
    args = parser.parse_args()
    return args


def get_vertex_coordinates(sequence_type):
    center = (0, 0)
    radius = 1
    vertex_coordinates = {}
    n_vertex_coordinates = 0
    vocab = None

    if sequence_type == "nucleotide":
        n_vertex_coordinates = 4
        vocab = "ACGT"
    elif sequence_type == "aminoacid":
        n_vertex_coordinates = 26
        vocab = "ARNDCQEGHILKMFPOSUTWYVBZXJ"

    angle = 2 * math.pi / n_vertex_coordinates
    for i in range(n_vertex_coordinates):
        vertex_coordinates[vocab[i]] = (center[0] + radius * math.cos(i * angle),
                                        center[1] + radius * math.sin(i * angle))
    return n_vertex_coordinates, vertex_coordinates


def generate_cgr(input_file, id_col, seq_col, sequence_type, output_dir):
    n_vertex_coordinates, vertex_coordinates = get_vertex_coordinates(sequence_type)
    print(f"Number of vertex coordinates = {n_vertex_coordinates}")
    print(f"Vertex coordinates = {vertex_coordinates}")

    cgr = CGR(n=n_vertex_coordinates, vertex_coordinates=vertex_coordinates)

    print(f"Input file = {input_file}")
    df = pd.read_csv(input_file)
    for i, row in df.iterrows():
        cgr.reset()
        output_filepath = os.path.join(output_dir, f"{row[id_col]}.png")
        cgr.encode(row[seq_col], output_filepath)


def main():
    config = parse_args()
    generate_cgr(config.input_file,
                 config.id_col,
                 config.seq_col,
                 config.sequence_type,
                 config.output_dir)
    return


if __name__ == "__main__":
    main()
    exit(0)
