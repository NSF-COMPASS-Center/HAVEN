import os
import pandas as pd
import numpy as np
from pathlib import Path
from utils import visualization_utils


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

        self.itr_col = "itr"
        self.y_true_col = "y_true"
        self.itrs = df["itr"].unique()

    def execute(self):
        if self.evaluation_settings["auroc"]:
            self.auroc()
        if self.evaluation_settings["auprc"]:
            self.auprc()
        if self.evaluation_settings["accuracy"]:
            self.accuracy()
        if self.evaluation_settings["f1"]:
            self.f1()
        if self.evaluation_settings["prediction_distribution"]:
            self.prediction_distribution()
        return

    def accuracy(self):
        result = []
        for itr in self.itrs:
            df_itr = self.df[self.df[self.itr_col] == itr]
            acc_itr = self.compute_accuracy(df_itr)
            result.append({self.itr_col: itr, "accuracy": acc_itr})
        result_df = pd.DataFrame(result)
        result_df.to_csv(self.evaluation_output_file_path + "accuracy.csv")
        visualization_utils.box_plot(result_df, "accuracy", self.visualization_output_file_path + "accuracy_boxplot.png")
        return

    def f1(self):
        result = []
        for itr in self.itrs:
            df_itr = self.df[self.df[self.itr_col] == itr]
            f1_itr = self.compute_f1(df_itr)
            result.append({self.itr_col: itr, "f1": f1_itr})
        result_df = pd.DataFrame(result)
        result_df.to_csv(self.evaluation_output_file_path + "f1.csv")
        visualization_utils.box_plot(result_df, "f1", self.visualization_output_file_path + "f1_boxplot.png")
        return

    def auroc(self):
        result = []
        for itr in self.itrs:
            df_itr = self.df[self.df[self.itr_col] == itr]
            auroc_itr = self.compute_auroc(df_itr)
            result.append({self.itr_col: itr, "auroc": auroc_itr})
        result_df = pd.DataFrame(result)
        result_df.to_csv(self.evaluation_output_file_path + "auroc.csv")
        visualization_utils.box_plot(result_df, "auroc", self.visualization_output_file_path + "auroc_boxplot.png")
        return

    def auprc(self):
        if self.evaluation_settings["type"] == "multi":
            print("ERROR: AUPRC not supported for multiclass classification.")
            return
        result = []
        for itr in self.itrs:
            df_itr = self.df[self.df[self.itr_col] == itr]
            auprc_itr = self.compute_auprc(df_itr)
            result.append({self.itr_col: itr, "auprc": auprc_itr})
        result_df = pd.DataFrame(result)
        result_df.to_csv(self.evaluation_output_file_path + "auprc.csv")
        visualization_utils.box_plot(result_df, "auprc", self.visualization_output_file_path + "auprc_boxplot.png")
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
                                                    self.visualization_output_file_path + "class_distribution.png")

    def compute_accuracy(self, df_itr) -> float:
        pass

    def compute_f1(self, df_itr) -> float:
        pass

    def compute_auroc(self, df_itr) -> float:
        pass

    def compute_auprc(self, df_itr) -> float:
        pass