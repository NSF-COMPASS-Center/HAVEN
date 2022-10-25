import argparse
import pandas as pd

from TransferModel.Analysis import metrics_computation
import TransferModel.DataUtils.DataProcessor as DataProcessor


def parse_args():
    parser = argparse.ArgumentParser(description='HEP Sequence Analysis Evaluation Pipeline')
    parser.add_argument('-d', '--data', required=True,
                        help="Comma separated list of absolute paths of csv files with prediction results.\n")
    parser.add_argument("-m", "--metrics", required=True, help="Comma separated list of metrics to be computed. Accepted values = acc,pr,rec,pr-curve,roc,auprc,auroc.\n")
    parser.add_argument("-t", "--targets", required=True, help="Comma separated list of prediction targets/labels.\n")
    parser.add_argument("-dir", "--output_dir", required=True, help="Absolute path to the output directory to store visualizations.\n")
    parser.add_argument("-ct", "--classification_type", required=True, help="Type of classification. Accepted values = binary, multi.\n")
    parser.add_argument("-op", "--output_prefix", required=True, help="Prefix of the generate output files\n")
    args = parser.parse_args()
    return args


def create_combined_data_df(data_files):
    create_combined_data_df_list = [pd.read_csv(data_file) for data_file in data_files]
    return pd.concat(create_combined_data_df_list, ignore_index=True)


def main():
    args = parse_args()
    data_files = args.data.split(",")
    metrics = args.metrics.split(",")
    targets = args.targets.split(",")
    classification_type = args.classification_type

    print("Output files = ", data_files)
    print("Metrics = ", metrics)
    print("Targets = ", targets)

    combined_output_df = create_combined_data_df(data_files)
    print("combined_output_df shape = ", combined_output_df.shape)
    print(combined_output_df)

    target_mapping_name_idx = DataProcessor.initilizeVocab(targets)
    target_mapping_name_idx['not-in-dictionary'] = 0
    print("target_mapping_name_idx = ", target_mapping_name_idx)

    target_mapping_idx_name = {v: k for k,v in target_mapping_name_idx.items()}
    print("target_mapping_idx_name = ", target_mapping_idx_name)

    model_values = combined_output_df["model"].unique()
    seed_values = combined_output_df["seed"].unique()

    if classification_type == "binary":
        metrics_computation.compute_metrics_binary(metrics, combined_output_df, target_mapping_idx_name, model_values, seed_values, args.output_dir, args.output_prefix)
    elif classification_type == "multi":
        metrics_computation.compute_metrics_multi(metrics, combined_output_df, target_mapping_idx_name, model_values,
                                                  seed_values, args.output_dir, args.output_prefix)
    else:
        print(f"ERROR: unsupported classification type: {classification_type}")


if __name__ == '__main__':
    main()
