from evaluation.evaluation_base import EvaluationBase
from sklearn.metrics import average_precision_score, accuracy_score, f1_score, roc_curve, auc, precision_recall_curve
import pandas as pd
from utils import visualization_utils


class BinaryClassEvaluation(EvaluationBase):
    def __init__(self, df, evaluation_settings, evaluation_output_file_base_path, visualization_output_file_base_path, output_file_name):
        super().__init__(df, evaluation_settings, evaluation_output_file_base_path, visualization_output_file_base_path, output_file_name)
        self.y_pred_column = self.evaluation_settings["positive_label"]

    def compute_accuracy(self, df_itr):
        y_pred = self.convert_probability_to_prediction(df_itr)
        return accuracy_score(y_true=df_itr[self.y_true_col].values, y_pred=y_pred)

    def compute_f1(self, df_itr):
        y_pred = self.convert_probability_to_prediction(df_itr)
        return f1_score(y_true=df_itr[self.y_true_col].values, y_pred=y_pred, pos_label=self.y_pred_column)

    def compute_auroc(self, df_itr):
        # The function roc_auc_score returns {1 - true_AUROC_score}
        # It considers the compliment of the prediction probabilities in the computation of the area
        # roc_auc_score(y_true=df_itr[self.y_true_col].values, y_score=df_itr["Human"].values)

        # Hence we use roc_curve to compute fpr, tpr followed by auc to compute the AUROC.
        fpr, tpr, _ = roc_curve(y_true=df_itr[self.y_true_col].values, y_score=df_itr[self.y_pred_column].values, pos_label=self.y_pred_column)
        return pd.DataFrame({"fpr": fpr, "tpr": tpr}), auc(fpr, tpr)

    def compute_auprc(self, df_itr):
        auprc_score = average_precision_score(y_true=df_itr[self.y_true_col].values, y_score=df_itr[self.y_pred_column].values, pos_label=self.y_pred_column)
        precision, recall, _ = precision_recall_curve(y_true=df_itr[self.y_true_col].values, probas_pred=df_itr[self.y_pred_column].values, pos_label=self.y_pred_column)
        return pd.DataFrame({"precision": precision, "recall": recall}), auprc_score

    def convert_probability_to_prediction(self, df_itr, threshold=0.5):
        y_pred_prob = df_itr[self.y_pred_column].values
        y_pred = [self.y_pred_column if y >= threshold else "Not " + self.y_pred_column for y in y_pred_prob]
        return y_pred

    def plot_visualizations(self):
        super()
        if self.evaluation_settings["auroc"]:
            visualization_utils.box_plot(self.evaluation_metrics_df, self.experiment_col, "auroc", self.visualization_output_file_path + "_auroc_boxplot.png")
            visualization_utils.curve_plot(df=self.roc_curves_df, x_col="fpr", y_col="tpr", color_group_col=self.experiment_col, style_group_col=self.itr_col, output_file_path=self.visualization_output_file_path + "_roc_curves.png")
        if self.evaluation_settings["auprc"]:
            visualization_utils.box_plot(self.evaluation_metrics_df, self.experiment_col, "auprc", self.visualization_output_file_path + "_auprc_boxplot.png")
            visualization_utils.curve_plot(df=self.pr_curves_df, x_col="recall", y_col="precision",
                                           color_group_col=self.experiment_col, style_group_col=self.itr_col,
                                           output_file_path=self.visualization_output_file_path + "_precision_recall_curves.png")
        return
