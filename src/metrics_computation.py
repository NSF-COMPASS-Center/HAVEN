from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import os
import pandas as pd

import metric_visualization


def compute_metrics(metrics, combined_output_df, target_mapping_idx_name, seed_values, output_dir):
    target_values = list(target_mapping_idx_name.keys())
    target_values.remove(0)
    target_values = [str(v) for v in target_values]
    for metric in metrics:
        if metric == "acc":
            compute_accuracy(combined_output_df, target_mapping_idx_name, seed_values, output_dir, target_values)
        # elif metric == "pr":
        #     compute_precision(combined_output_df, target_mapping_idx_name, seed_values)
        # elif metric == "rec":
        #     compute_recall(combined_output_df, target_mapping_idx_name, seed_values)
        # elif metric == "pr":
        #     compute_precision_recall_curve(combined_output_df, target_mapping_idx_name, seed_values)
        # elif metric == "roc":
        #    compute_roc(combined_output_df, target_mapping_idx_name, seed_values)
        elif metric == "auprc":
            compute_auprc(combined_output_df, target_mapping_idx_name, seed_values, output_dir, target_values)
            metric_visualization.binary_precision_recall_curves(combined_output_df, y_test_col="test_label", y_pred_col="1", itr_col="seed", output_file_path=os.path.join(output_dir, "precision_recall_curves.png"))
        elif metric == "auroc":
            compute_auroc(combined_output_df, target_mapping_idx_name, seed_values, output_dir, target_values)
            metric_visualization.binary_roc_curves(combined_output_df, y_test_col="test_label", y_pred_col="1", itr_col="seed", output_file_path=os.path.join(output_dir, "roc_curves.png"))
        else:
            print(f"ERROR: Unsupported metric {metric}")
    return


def compute_accuracy(combined_output_df, target_mapping_idx_name, seed_values, output_dir, target_values):
    accuracy_summary = []
    for seed_value in seed_values:
        df = combined_output_df[combined_output_df.seed == seed_value]
        accuracy = {"metric": "accuracy", "seed": seed_value}
        for target in target_values:
            combined_output_df[target] = [1 if y >= 0.5 else 0 for y in combined_output_df[target]]
            target_accuracy = accuracy_score(y_true=df["test_label"], y_pred=combined_output_df[target])
            accuracy["target"] = target_mapping_idx_name[int(target)]
            accuracy["metric_val"] = target_accuracy
        accuracy_summary.append(accuracy)
    df = pd.DataFrame(accuracy_summary)
    metric_visualization.box_plot(df, os.path.join(output_dir, "accuracy_box_plot.png"), "target", "metric_val", "accuracy")


def compute_auprc(combined_output_df, target_mapping_idx_name, seed_values, output_dir, target_values):
    auprc_summary = []
    for seed_value in seed_values:
        df = combined_output_df[combined_output_df.seed == seed_value]
        auprc = {"metric": "auprc", "seed": seed_value}
        for target in target_values:
            target_auprc = average_precision_score(y_true=df["test_label"], y_score=combined_output_df[target])
            auprc["target"] = target_mapping_idx_name[int(target)]
            auprc["metric_val"] = target_auprc
        auprc_summary.append(auprc)
    df = pd.DataFrame(auprc_summary)
    metric_visualization.box_plot(df, os.path.join(output_dir, "auprc_box_plot.png"), "target", "metric_val", "auprc")


def compute_auroc(combined_output_df, target_mapping_idx_name, seed_values, output_dir, target_values):
    auroc_summary = []
    for seed_value in seed_values:
        df = combined_output_df[combined_output_df.seed == seed_value]
        auroc = {"metric": "auroc", "seed": seed_value}
        for target in target_values:
            target_auroc = roc_auc_score(y_true=df["test_label"], y_score=combined_output_df[target])
            auroc["target"] = target_mapping_idx_name[int(target)]
            auroc["metric_val"] = target_auroc
        auroc_summary.append(auroc)
    df = pd.DataFrame(auroc_summary)
    metric_visualization.box_plot(df, os.path.join(output_dir, "auroc_box_plot.png"), "target", "metric_val", "auroc")

