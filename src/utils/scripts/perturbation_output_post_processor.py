#!/usr/src/env python
import argparse
import ast

import yaml
import os
import pandas as pd
from pathlib import Path
import re

perturb_pos_col = "perturb_pos"
orig_token_col = "orig_token"
new_token_col = "new_token"
temp_col = "temp_col"
id_parser_regex_pattern = re.compile("(.+)_([A-Z])_(\d+)_([A-Z])")

def parse_args():
    parser = argparse.ArgumentParser(description='Post process the output of perturbated dataset prediction')
    parser.add_argument("-id", "--input_dir", required=True,
                        help="Absolute path to input directory with all the prediction output files.\n")
    parser.add_argument("-od", "--output_dir", required=True,
                        help="Absolute path to output directory.\n")
    parser.add_argument("-id_col", "--id_col", required=True,
                        help="Name of the id column in the prediction output files. Examples: 'uniprot_id', 'uniref90_id'.\n")
    args = parser.parse_args()
    return args

# parse id columns of the format <id>_<orig token>_<perturb pos>_<new token>
# eg: UniRef90_A0A7T6Y5W2_X_283_V, WIV04_E_323_A
# returns id, orig_token, perturb_pos, new_token
def parse_id(id_val):
    match_result = id_parser_regex_pattern.match(id_val)
    if match_result:
        id, orig_token, perturb_pos, new_token = match_result.group(1, 2, 3, 4)
        return id, orig_token, perturb_pos, new_token
    else:
        return None


def post_process_output(input_dir, output_dir, id_col):
    input_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    processing_error = []
    for input_file in input_files:
        print(f"{input_file}")
        df = pd.read_csv(os.path.join(input_dir, input_file), converters={id_col: ast.literal_eval})
        df[id_col] = df[id_col].map(lambda x: x.pop())

        # assuming the id follows the pattern of <alphanumeric id>_<orig token>_<perturb pos>_<new token>
        # sequences that do not follow this id pattern will error out in the next line.
        try:
            df[[id_col, orig_token_col, perturb_pos_col, new_token_col]] = df.apply(lambda x: parse_id(x[id_col]), axis=1, result_type="expand")
        except:
            processing_error.append(input_file)
            continue

        output_file_name = os.path.basename(input_file)
        # create any missing parent directories
        output_file_path = os.path.join(output_dir, output_file_name)
        Path(os.path.dirname(output_file_path)).mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file_path, index=False)

    if len(processing_error) > 0:
        print(f"ERROR: Skipped processing following {len(processing_error)} files due to filename not conforming to the expected sequence id pattern:")
        print("\n".join(processing_error))


def main():
    config = parse_args()
    input_dir = config.input_dir
    output_dir = config.output_dir
    id_col = config.id_col
    post_process_output(input_dir, output_dir, id_col)
    return


if __name__ == '__main__':
    main()
    exit(0)
