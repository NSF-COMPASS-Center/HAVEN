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
    input_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    processing_error = []
    for input_file in input_files:
        print(f"{input_file}")
        df = pd.read_csv(os.path.join(input_dir, input_file), converters={id_col: ast.literal_eval})
        df[id_col] = df[id_col].map(lambda x: x.pop())

        # assuming the id follows the pattern of id_<origtoken>_<perturbpos>_<newtoken>
        # sequences that do not follow this id pattern will error out in the next line.
        # tactical fix: record the filename, and continue
        # TODO: implement a strategical fix
        try:
            df[[id_col, orig_token_col, perturb_pos_col, new_token_col]] = df[id_col].str.split("_", expand=True)
        except:
            processing_error.append(input_file)
            continue

        output_file_name = os.path.basename(input_file)
        # create any missing parent directories
        output_file_path = os.path.join(output_dir, output_file_name)
        Path(os.path.dirname(output_file_path)).mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file_path, index=False)

    print(f"ERROR: Skipped processing following {len(processing_error)} files due to filename not conforming to the expected sequence id pattern:")
    print("\n".join(processing_error))


def main():
    config = parse_args()
    input_dir = config.input_dir
    output_dir = config.output_dir
    post_process_output(input_dir, output_dir)
    return


if __name__ == '__main__':
    main()
    exit(0)
