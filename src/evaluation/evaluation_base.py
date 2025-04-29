import os
import pandas as pd
import numpy as np
from pathlib import Path
from utils import utils, visualization_utils


class EvaluationBase:
    def __init__(self, df, evaluation_settings, evaluation_output_file_base_path, visualization_output_file_base_path, output_file_name):
        self.df = df
        self.evaluation_settings = evaluation_settings
        self.evaluation_output_file_base_path = evaluation_output_file_base_path
        self.visualization_output_file_base_path = visualization_output_file_base_path
        self.output_file_name = output_file_name

        self.evaluation_output_file_path = os.path.join(evaluation_output_file_base_path, output_file_name)
        Path(os.path.dirname(self.evaluation_output_file_path)).mkdir(parents=True, exist_ok=True)

        self.visualization_output_file_path = os.path.join(visualization_output_file_base_path, output_file_name)
        Path(os.path.dirname(self.visualization_output_file_path)).mkdir(parents=True, exist_ok=True)

        self.evaluation_metrics_df = None
        self.roc_curves_df = None
        self.pr_curves_df = None
        self.metadata = None
        self.itr_col = "itr"
        self.experiment_col = "experiment"
        self.y_true_col = "y_true"
        self.itrs = df[self.itr_col].unique()

    def execute(self):
        experiments = self.df[self.experiment_col].unique()
        result = []
        roc_curves = []
        pr_curves = []

        for experiment in experiments:
            experiment_df = self.df[self.df[self.experiment_col] == experiment]
            print(self.df.head())  # Ensure it's not empty
            print(self.df.columns)
            for itr in self.itrs:
                result_itr = {self.itr_col: itr, self.experiment_col: experiment}
                df_itr = experiment_df[experiment_df[self.itr_col] == itr]
                try:
                    if self.metadata is None:
                        # compute the metadata only once
                        self.metadata = utils.compute_class_distribution(df_itr, self.y_true_col, format=True)
                    if self.evaluation_settings["auroc"]:
                        roc_curve_itr, auroc_itr = self.compute_auroc(df_itr)
                        # individual ROC curves
                        roc_curve_itr[self.itr_col] = itr
                        roc_curve_itr[self.experiment_col] = experiment
                        roc_curves.append(roc_curve_itr)
                        result_itr["auroc"] = auroc_itr
                    if self.evaluation_settings["auprc"]:
                        pr_curve_itr, auprc_itr = self.compute_auprc(df_itr)
                        # individual Precision-Recall curves
                        pr_curve_itr[self.itr_col] = itr
                        pr_curve_itr[self.experiment_col] = experiment
                        pr_curves.append(pr_curve_itr)
                        result_itr["auprc"] = auprc_itr
                    if self.evaluation_settings["accuracy"]:
                        acc_itr = self.compute_accuracy(df_itr)
                        result_itr["accuracy"] = acc_itr
                    if self.evaluation_settings["f1"]:
                        f1_itr = self.compute_f1(df_itr)
                        result_itr["f1"] = f1_itr
                    if self.evaluation_settings["prediction_distribution"]:
                        self.prediction_distribution()
                    result.append(result_itr)
                except Exception as e:
                    print(e)
                    pass
        self.evaluation_metrics_df = pd.DataFrame(result)
        print(self.evaluation_metrics_df.head())
        self.evaluation_metrics_df.to_csv(self.evaluation_output_file_path + "_evaluation_metrics.csv")

        if len(roc_curves) > 0:
            self.roc_curves_df = pd.concat(roc_curves, ignore_index=True)
            self.roc_curves_df.to_csv(self.evaluation_output_file_path + "_roc_curves.csv")
        if len(pr_curves) > 0:
            self.pr_curves_df = pd.concat(pr_curves, ignore_index=True)
            self.pr_curves_df.to_csv(self.evaluation_output_file_path + "_pr_curves.csv")

        self.plot_visualizations()
        return

    def plot_visualizations(self):
        if self.evaluation_settings["accuracy"]:
            visualization_utils.box_plot(self.evaluation_metrics_df, self.experiment_col, "accuracy", self.visualization_output_file_path + "_accuracy_boxplot.pdf")
        if self.evaluation_settings["f1"]:
            visualization_utils.box_plot(self.evaluation_metrics_df, self.experiment_col, "f1", self.visualization_output_file_path + "_f1_boxplot.pdf")
        if self.evaluation_settings["prediction_distribution"]:
            self.prediction_distribution()
        return

    def prediction_distribution(self):
        result = []
        for itr in self.itrs:
            df_itr = self.df[self.df[self.itr_col] == itr]
            y_pred = self.convert_probability_to_prediction(df_itr)
            labels, counts = np.unique(y_pred, return_counts=True)
            result_itr_pred = pd.DataFrame({"label": labels, "label_count": counts, "itr": itr, "group": "y_pred"})

            y_true = df_itr[self.y_true_col].values
            labels, counts = np.unique(y_true, return_counts=True)
            result_itr_true = pd.DataFrame({"label": labels, "label_count": counts, "itr": itr, "group": "y_true"})

            result.append(result_itr_pred)
            result.append(result_itr_true)
        result_df = pd.concat(result)
        result_df.to_csv(self.evaluation_output_file_path + "class_distribution.csv")
        visualization_utils.class_distribution_plot(result_df,
                                                    self.visualization_output_file_path + "class_distribution.pdf")

    def compute_accuracy(self, df_itr) -> float:
        pass

    def compute_f1(self, df_itr) -> float:
        pass

    def compute_auroc(self, df_itr) -> float:
        pass

    def compute_auprc(self, df_itr) -> float:
        pass