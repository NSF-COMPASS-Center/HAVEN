from evaluation.EvaluationBase import EvaluationBase
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


class MultiClassEvaluation(EvaluationBase):
    def __init__(self, df, evaluation_settings, evaluation_output_file_base_path, visualization_output_file_base_path,
                 output_file_name):
        super().__init__(df, evaluation_settings, evaluation_output_file_base_path, visualization_output_file_base_path,
                         output_file_name)
        self.y_pred_columns = self.get_y_pred_columns()

    def get_y_pred_columns(self):
        y_pred_columns = list(self.df.columns.values)
        y_pred_columns.remove(self.itr_col)
        y_pred_columns.remove(self.y_true_col)
        y_pred_columns.remove(self.experiment_col)
        return y_pred_columns

    def compute_accuracy(self, df_itr):
        y_pred = self.convert_probability_to_prediction(df_itr)
        return accuracy_score(y_true=df_itr[self.y_true_col].values, y_pred=y_pred)

    def compute_f1(self, df_itr):
        y_pred = self.convert_probability_to_prediction(df_itr)
        return f1_score(y_true=df_itr[self.y_true_col].values, y_pred=y_pred, average="macro")

    def compute_auroc(self, df_itr):
        return roc_auc_score(y_true=df_itr[self.y_true_col], y_score=df_itr[self.y_pred_columns], multi_class="ovr")

    def compute_auprc(self, df_itr):
        # print("ERROR: AUPRC not supported for multiclass classification.")
        pass

    def convert_probability_to_prediction(self, df_itr):
        y_pred_prob = df_itr[self.y_pred_columns]
        return [y for y in y_pred_prob.idxmax(axis="columns")]
