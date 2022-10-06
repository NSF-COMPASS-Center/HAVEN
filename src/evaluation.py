import argparse
import pandas as pd

import metrics_computation
import TransferModel.DataUtils.DataProcessor as DataProcessor


def parse_args():
    parser = argparse.ArgumentParser(description='HEP Sequence Analysis Evaluation Pipeline')
    parser.add_argument('-i', '--inputs', required=True,
                        help="Comma separated list of absolute paths of csv files with prediction results.\n")
    parser.add_argument("-m", "--metrics", required=True, help="Comma separated list of metrics to be computed. Accepted values = acc,pr,rec,pr-curve,roc,auprc,auroc")
    parser.add_argument("-t", "--targets", required=True, help="Comma separated list of prediction targets/labels")
    parser.add_argument("-dir", "--output_dir", required=True, help="Absolute path to the output directory to store visualizations.")
    args = parser.parse_args()
    return args


def create_combined_input_df(input_files):
    create_combined_input_df_list = [pd.read_csv(input_file) for input_file in input_files]
    return pd.concat(create_combined_input_df_list, ignore_index=True)


def main():
    args = parse_args()
    input_files = args.inputs.split(",")
    metrics = args.metrics.split(",")
    targets = args.targets.split(",")

    print("Output files = ", input_files)
    print("Metrics = ", metrics)
    print("Targets = ", targets)

    combined_output_df = create_combined_input_df(input_files)
    print("combined_output_df shape = ", combined_output_df.shape)
    print(combined_output_df)

    target_mapping_name_idx = DataProcessor.initilizeVocab(targets)
    target_mapping_name_idx['not-in-dictionary'] = 0
    print("target_mapping_name_idx = ", target_mapping_name_idx)

    target_mapping_idx_name = {v: k for k,v in target_mapping_name_idx.items()}
    print("target_mapping_idx_name = ", target_mapping_idx_name)

    seed_values = combined_output_df["seed"].unique()
    metrics_computation.compute_metrics(metrics, combined_output_df, target_mapping_idx_name, seed_values, args.output_dir)


if __name__ == '__main__':
    main()
