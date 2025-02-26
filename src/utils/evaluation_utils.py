from sklearn.metrics import auc, precision_recall_curve, f1_score
import pandas as pd
import torch


# compute class-wise auprc in given df
def compute_class_auprc(df, y_pred_columns, y_true_col):
    pr_curves = []
    auprcs = []
    for y_pred_column in y_pred_columns:
        precision, recall, _ = precision_recall_curve(y_true=df[y_true_col].values, probas_pred=df[y_pred_column].values, pos_label=y_pred_column)
        pr_curves.append(pd.DataFrame({"precision": precision, "recall": recall, "class": y_pred_column}))
        auprcs.append({"class": y_pred_column, "auprc": auc(recall, precision)})
    return pd.concat(pr_curves, ignore_index=True), pd.DataFrame(auprcs)


def get_f1_score(y_true, y_pred, select_non_zero=False):
    # convert the y_pred from b x n_classes x n to b x n x n_classes
    y_pred = y_pred.transpose(1, 2)

    # compute softmax over the predictions
    y_pred = torch.nn.functional.softmax(y_pred, dim=-1)

    # select the class index with the highest prediction as the class prediction for that sequence
    y_pred_vals, y_pred_index = torch.max(y_pred, dim=2)
    y_pred = y_pred_index

    if select_non_zero:
        non_zero_indices = torch.nonzero(y_true, as_tuple=True)
        y_true = y_true[non_zero_indices]
        y_pred = y_pred[non_zero_indices]

    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    return f1_micro, f1_macro

