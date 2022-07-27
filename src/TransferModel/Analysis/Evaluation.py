from TransferModel.Analysis import Visualization as Visualization
import TransferModel.DataUtils.DataProcessor as DataProcessor
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score


# y_test and y_pred are assumed to be in dense and sparse formats respectively for user convenience.

def report_auroc(y_test, y_pred, labelVocab=None, filename=None, average=None, multi_class="ovr"):
    """
    Returns auroc score(s)
    Args:
        y_test: reference y values to model in dense/non-sparse format, e.g [1,2] for classes 1 and 2
        y_pred: Outputs of model in sparse format
        labelVocab: dictionary of labels, e.g {"Human": 1, "Unknown": 0} (Can be None if not writing to file)
        filename: file to write to (if wanting to write to disk)
        average: averaging method (See scikitlearn roc_auc_score)
        multi_class: multi class method (See scikitlearn roc_auc_score)

    Returns: Array of roc scores per class if average=None, or a single roc score if average is set
    """
    assert labelVocab

    y_test = DataProcessor.toOneHot(y_test, labelVocab)

    if filename:
        Visualization.plot_auroc(y_test, y_pred, labelVocab, filename)

    return roc_auc_score(y_test, y_pred, average=average, multi_class=multi_class)


def report_accuracy_per_class(y_test, y_pred, yDict):
    """
    Returns accuracies
    We define accuracy for a class as (TP+TN)/(TP+TN+FP+FN) over every time the model predicted the class.
    i.e Everytime the model predicts class "a," how often is this prediction correct (tp or tn)?

    Args:
        y_test: reference y values in dense format
        y_pred: Outputs of model in sparse format
        yDict: Dictionary mapping string label to integer classes (to ensure correct ordering of matrix)

    Returns: Numpy array of accuracy per class in sorted order class 0, ... n
    """
    y_pred = DataProcessor.sparseToDense(y_pred)
    matrix = confusion_matrix(y_test, y_pred, labels=sorted(list(yDict.values())))
    return matrix.diagonal() / matrix.sum(axis=1)


def report_accuracy(y_test, y_pred):
    """
    Returns the accuracy
    We define accuracy as standard: TP+TN/(TP+TN+FP+FN)
    Args:
        y_test: reference y values in dense format
        y_pred: Outputs of model in sparse format
    Returns: Accuracy percentage from [0,1]
    """
    y_pred = DataProcessor.sparseToDense(y_pred)
    return accuracy_score(y_test, y_pred)
