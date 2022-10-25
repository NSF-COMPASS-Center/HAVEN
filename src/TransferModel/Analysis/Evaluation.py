from collections import defaultdict, Counter

import pandas as pd

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


def report_confusion_matrix(y_test, y_pred, yDict):
    """
    Args:
        y_test: reference y values in dense format
        y_pred: Outputs of model in sparse format
        yDict: Dictionary mapping string label to integer classes (to ensure correct ordering of matrix)
    Returns:
        SKL confusion matrix with rows being true label and cols being predicted labels
    """
    y_pred = DataProcessor.sparseToDense(y_pred)
    matrix = confusion_matrix(y_test, y_pred, labels=sorted(list(yDict.values())))
    return matrix


def report_class_distribution(y_test, y_pred, yDict, norm=0, matrix=None):
    """
       Returns how frequently the model returns each class

       Args:
           y_test: r
           y_pred: Outputs of model in sparse format
           yDict: Dictionary mapping string label to integer classes (to ensure correct ordering of matrix)
           norm: 0 == Frequency of model; 1 ==  Frequency of dataset
           matrix: Confusion matrix given by user. Will generate from y_test, y_pred, yVocab if not given.

       Returns: Numpy array of predicted distribution
    """

    if matrix is None:
        matrix = report_confusion_matrix(y_test, y_pred, yDict)

    tmp = matrix.sum(axis=((1 + norm) % 2))
    tmp = tmp / tmp.sum(axis=0)
    return tmp, matrix


def report_accuracy_per_class(y_test, y_pred, yDict, matrix=None):
    """
    Returns accuracies normalized over the test set
    Accuracy of class 'a' is (Tp_a + Tn_a) / (TP_a + TN_a + FP_a + FN_a)

    Args:
        y_test: reference y values in dense format
        y_pred: Outputs of model in sparse format
        yDict: Dictionary mapping string label to integer classes (to ensure correct ordering of matrix)
        matrix: Confusion matrix given by user. Will generate from y_test, y_pred, yVocab if not given.

    Returns: Numpy array of accuracy per class in sorted order class 0, ... n, generated confusion matrix
    """
    if matrix is None:
        matrix = report_confusion_matrix(y_test, y_pred, yDict)
    return matrix.diagonal() / (matrix.sum(axis=0) + matrix.sum(axis=1) - matrix.diagonal()), matrix


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


def report_cluster_purity(adata):
    """

    Args:
        adata: AnnData frame for scanpy. Must have already called sc.tl.louvian

    Returns:
    """
    # Report purity for each embed target using louvian method
    groupedEntries = adata.obs.groupby("louvain")

    for category in adata.obs.columns:
        if category is not "louvain":
            print(f"------------------- Category: {category} ---------------------")
            # Calculate purity by category within this cluster
            categoryMaps = groupedEntries[category].agg(Counter)
            for louGroup in groupedEntries.groups:
                # louGroup, category
                freqMap = categoryMaps[louGroup]
                mostCommon = freqMap.most_common(1)[0]
                purity = mostCommon[1] / sum(freqMap.values())
                print(f"Purity of louvain cluster {louGroup} in category {category} with majority element {mostCommon[0]}: {purity}")









