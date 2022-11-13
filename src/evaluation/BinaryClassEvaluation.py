from evaluation.EvaluationBase import EvaluationBase
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score


class BinaryClassEvaluation(EvaluationBase):
    def __init__(self, df, evaluation_settings, evaluation_output_file_base_path, visualization_output_file_base_path, output_file_name):
        super().__init__(df, evaluation_settings, evaluation_output_file_base_path, visualization_output_file_base_path, output_file_name)
        self.y_pred_column = "Human"

    def compute_accuracy(self, df_itr):
        y_pred = self.convert_probability_to_prediction(df_itr)
        return accuracy_score(y_true=df_itr[self.y_true_col].values, y_pred=y_pred)

    def compute_f1(self, df_itr):
        y_pred = self.convert_probability_to_prediction(df_itr)
        return f1_score(y_true=df_itr[self.y_true_col].values, y_pred=y_pred, pos_label="Human")

    def compute_auroc(self, df_itr):
        return roc_auc_score(y_true=df_itr[self.y_true_col].values, y_score=df_itr["Human"].values, average="macro")

    def compute_auprc(self, df_itr):
        return average_precision_score(y_true=df_itr[self.y_true_col].values, y_score=df_itr["Human"].values, average="macro", pos_label="Human")

    def convert_probability_to_prediction(self, df_itr, threshold=0.5):
        y_pred_prob = df_itr[self.y_pred_column].values
        y_pred = ["Human" if y >= threshold else "Not Human" for y in y_pred_prob]
        return y_pred
