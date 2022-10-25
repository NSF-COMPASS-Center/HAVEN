from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import os
import pandas as pd
import numpy as np

from TransferModel.Analysis import metrics_visualization


def compute_metrics_binary(metrics, combined_output_df, target_mapping_idx_name, model_values, seed_values, output_dir, output_prefix):
    target_values = list(target_mapping_idx_name.keys())
    target_values.remove(0)
    target_values = [str(v) for v in target_values]
    for metric in metrics:
        if metric == "acc":
            compute_accuracy_binary(combined_output_df, target_mapping_idx_name, model_values, seed_values, output_dir, target_values, output_prefix)
        # elif metric == "pr":
        #     compute_precision(combined_output_df, target_mapping_idx_name, seed_values)
        # elif metric == "rec":
        #     compute_recall(combined_output_df, target_mapping_idx_name, seed_values)
        elif metric == "pr-curve":
            metrics_visualization.binary_precision_recall_curves(combined_output_df, y_test_col="test_label", y_pred_col="1", itr_col="seed", output_file_path=os.path.join(output_dir, "precision_recall_curves.png"))
        elif metric == "roc":
            metrics_visualization.binary_roc_curves(combined_output_df, y_test_col="test_label", y_pred_col="1", itr_col="seed", output_file_path=os.path.join(output_dir, "roc_curves.png"))
        elif metric == "auprc":
            compute_auprc_binary(combined_output_df, target_mapping_idx_name, model_values, seed_values, output_dir, target_values, output_prefix)
        elif metric == "auroc":
            compute_auroc_binary(combined_output_df, target_mapping_idx_name, model_values, seed_values, output_dir, target_values, output_prefix)
        else:
            print(f"ERROR: Unsupported metric {metric}")
    return


def compute_metrics_multi(metrics, combined_output_df, target_mapping_idx_name, model_values, seed_values, output_dir, output_prefix):
    target_values = list(target_mapping_idx_name.keys())
    target_values = [str(v) for v in target_values]
    for metric in metrics:
        if metric == "acc":
            compute_accuracy_multi(combined_output_df, target_mapping_idx_name, model_values, seed_values, output_dir, target_values, output_prefix)
        # elif metric == "pr":
        #     compute_precision(combined_output_df, target_mapping_idx_name, seed_values)
        # elif metric == "rec":
        #     compute_recall(combined_output_df, target_mapping_idx_name, seed_values)
        elif metric == "pr-curve":
            metrics_visualization.binary_precision_recall_curves(combined_output_df, y_test_col="test_label", y_pred_col="1", itr_col="seed", output_file_path=os.path.join(output_dir, "precision_recall_curves.png"))
        elif metric == "roc":
            metrics_visualization.binary_roc_curves(combined_output_df, y_test_col="test_label", y_pred_col="1", itr_col="seed", output_file_path=os.path.join(output_dir, "roc_curves.png"))
        elif metric == "auprc":
            compute_auprc_multi(combined_output_df, target_mapping_idx_name, model_values, seed_values, output_dir, target_values, output_prefix)
        elif metric == "auroc":
            compute_auroc_multi(combined_output_df, target_mapping_idx_name, model_values, seed_values, output_dir, target_values, output_prefix)
        else:
            print(f"ERROR: Unsupported metric {metric}")
    return


def compute_accuracy_multi(combined_output_df, target_mapping_idx_name, model_values, seed_values, output_dir, target_values, output_prefix):
    accuracy_summary = []
    # combined_output_df = combined_output_df.astype({"test_label": "int32"})
    for model_value in model_values:
        model_df = combined_output_df[combined_output_df.model == model_value]
        for seed_value in seed_values:
            df = model_df[model_df.seed == seed_value]
            accuracy = {"metric": "accuracy", "model": model_value, "seed": seed_value}
            df["y_pred"] = df[target_values].idxmax(axis="columns")
            df = df.astype({"y_pred": "float"})
            accuracy_val = accuracy_score(y_true=df["test_label"], y_pred=df["y_pred"])
            print(accuracy_val)
            accuracy["metric_val"] = accuracy_val
            accuracy_summary.append(accuracy)
    df = pd.DataFrame(accuracy_summary)
    metrics_visualization.box_plot(df, os.path.join(output_dir, output_prefix + "_accuracy_box_plot.png"), "model", "metric_val", "accuracy", 0.60)


def compute_auprc_multi(combined_output_df, target_mapping_idx_name, model_values, seed_values, output_dir, target_values, output_prefix):
    auprc_summary = []
    for model_value in model_values:
        model_df = combined_output_df[combined_output_df.model == model_value]
        for seed_value in seed_values:
            df = model_df[model_df.seed == seed_value]
            y_true = convert_multiclass_label_to_binary(df["test_label"], target_values)
            auprc = {"metric": "auprc", "model": model_value, "seed": seed_value}
            auprc_val = average_precision_score(y_true=y_true, y_score=df[target_values])
            auprc["metric_val"] = auprc_val
            auprc_summary.append(auprc)
    df = pd.DataFrame(auprc_summary)
    metrics_visualization.box_plot(df, os.path.join(output_dir, output_prefix + "_auprc_box_plot.png"), "model", "metric_val", "macro-auprc")


def compute_auroc_multi(combined_output_df, target_mapping_idx_name, model_values, seed_values, output_dir, target_values, output_prefix):
    auroc_summary = []
    for model_value in model_values:
        model_df = combined_output_df[combined_output_df.model == model_value]
        for seed_value in seed_values:
            df = model_df[model_df.seed == seed_value]
            y_true = convert_multiclass_label_to_binary(df["test_label"], target_values)
            auroc = {"metric": "auroc", "model": model_value, "seed": seed_value}
            auroc_val = roc_auc_score(y_true=y_true, y_score=df[target_values])
            auroc["metric_val"] = auroc_val
            auroc_summary.append(auroc)
    df = pd.DataFrame(auroc_summary)
    metrics_visualization.box_plot(df, os.path.join(output_dir, output_prefix + "_auroc_box_plot.png"), "model", "metric_val", "macro-auroc", 0.11)


def convert_multiclass_label_to_binary(y, labels):
    n = len(y)
    y_bin = np.zeros((n, len(labels)))
    for idx, val in enumerate(y):
        y_bin[idx][int(val)] = 1
    return pd.DataFrame(y_bin, columns=labels)


def compute_accuracy_binary(combined_output_df, target_mapping_idx_name, model_values, seed_values, output_dir, target_values, output_prefix):
    accuracy_summary = []
    for model_value in model_values:
        model_df = combined_output_df[combined_output_df.model == model_value]
        for seed_value in seed_values:
            df = model_df[model_df.seed == seed_value]
            accuracy = {"metric": "accuracy", "model": model_value, "seed": seed_value}
            for target in target_values:
                df[target] = [1 if y >= 0.5 else 0 for y in df[target]]
                target_accuracy = accuracy_score(y_true=df["test_label"], y_pred=df[target])
                accuracy["target"] = target_mapping_idx_name[int(target)]
                accuracy["metric_val"] = target_accuracy
            accuracy_summary.append(accuracy)
    df = pd.DataFrame(accuracy_summary)
    metrics_visualization.box_plot(df, os.path.join(output_dir, output_prefix + "_accuracy_box_plot.png"), "model", "metric_val", "accuracy", 0.73)


def compute_auprc_binary(combined_output_df, target_mapping_idx_name, model_values, seed_values, output_dir, target_values, output_prefix):
    auprc_summary = []
    for model_value in model_values:
        model_df = combined_output_df[combined_output_df.model == model_value]
        for seed_value in seed_values:
            df = model_df[model_df.seed == seed_value]
            auprc = {"metric": "auprc", "model": model_value, "seed": seed_value}
            for target in target_values:
                target_auprc = average_precision_score(y_true=df["test_label"], y_score=df[target])
                auprc["target"] = target_mapping_idx_name[int(target)]
                auprc["metric_val"] = target_auprc
            auprc_summary.append(auprc)
    df = pd.DataFrame(auprc_summary)
    metrics_visualization.box_plot(df, os.path.join(output_dir, output_prefix + "_auprc_box_plot.png"), "model", "metric_val", "auprc", 0.73)


def compute_auroc_binary(combined_output_df, target_mapping_idx_name, model_values, seed_values, output_dir, target_values, output_prefix):
    auroc_summary = []
    for model_value in model_values:
        model_df = combined_output_df[combined_output_df.model == model_value]
        for seed_value in seed_values:
            df = model_df[model_df.seed == seed_value]
            auroc = {"metric": "auroc", "model": model_value, "seed": seed_value}
            for target in target_values:
                target_auroc = roc_auc_score(y_true=df["test_label"], y_score=df[target])
                auroc["target"] = target_mapping_idx_name[int(target)]
                auroc["metric_val"] = target_auroc
            auroc_summary.append(auroc)
    df = pd.DataFrame(auroc_summary)
    metrics_visualization.box_plot(df, os.path.join(output_dir, output_prefix + "_auroc_box_plot.png"), "model", "metric_val", "auroc", 0.5)

