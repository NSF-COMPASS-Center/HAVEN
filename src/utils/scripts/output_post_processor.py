#!/usr/src/env python
import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Post process the output of predictions')
    parser.add_argument("-if", "--input_file_template", required=True,
                        help="Absolute path to input file template with {itr} as placeholder for iteration number.\n")
    parser.add_argument("-of", "--output_file_path", required=True,
                        help="Absolute path to where the output file should be written.\n")
    parser.add_argument("-n_itrs", "--n_iterations", required=True,
                        help="Number of iterations.\n")
    args = parser.parse_args()
    return args

def aggregate_output(input_file_template, output_file_path, n_iterations):
    result_dfs = []
    for i in range(n_iterations):
        input_file_path = input_file_template.format(itr=i)
        result_dfs.append(pd.read_csv(input_file_path, index_col=0))
    pd.concat(result_dfs).to_csv(output_file_path, index=True)

def main():
    config = parse_args()
    aggregate_output(config.input_file_template, config.output_file_path, config.n_iterations)
    return


if __name__ == '__main__':
    main()
    exit(0)
