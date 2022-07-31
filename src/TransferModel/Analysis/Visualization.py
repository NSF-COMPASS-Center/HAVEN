import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import scanpy as sc

def plot_train_test_metric(trainMetric, testMetric, filename, metricName="Cross Entropy", optimalFlag="min"):
    """
    Used for plotting train and test metrics indexed by epochs to files
    Args:
        trainMetric: Array of test metric values indexed by epoch
        testMetric: Array of train metric values indexed by epoch
        filename: filename to write to
        metricName: y-axis name
        optimalFlag: plots a horizontal line depending on the optimal epoch choices are: ["min", "max", None]
    """
    assert filename
    with plt.style.context("seaborn"):
        fig = plt.figure(1, [16, 9])
        epochs = range(1, len(trainMetric) + 1)
        if trainMetric:
            plt.plot(epochs, trainMetric, label=f"Training {metricName}")
            if optimalFlag:
                optimal = min(trainMetric) if optimalFlag == 'min' else max(trainMetric)
                plt.hlines(y=optimal, xmin=0, xmax=len(trainMetric) + 1,
                           label=f"Training set optimum {metricName}: {optimal}", linestyles='-.', lw=1)
        if testMetric:
            plt.plot(epochs, testMetric, label=f"Testing {metricName}")
            if optimalFlag:
                optimal = min(testMetric) if optimalFlag == "min" else max(testMetric)
                plt.hlines(y=optimal, xmin=0, xmax=len(testMetric) + 1,
                           label=f"Testing set optimum {metricName}: {optimal}", linestyles='--', lw=1)

    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel(f"{metricName}", fontsize=20)
    plt.title('Training history', fontsize=20)
    plt.legend()

    # Print image
    name = f'figures/{filename}.png'
    plt.savefig(name, bbox_inches='tight', dpi=300)
    plt.clf()


def plot_metrics(dic, filename, metricName="AUROC"):
    """
    Plots metrics in figures/filename.png
    Args:
        dic: Dictionary of label:[data1, data2, ...] pairs
        filename: filename to write to
        metricName: y axis label
    """
    with plt.style.context("seaborn"):
        fig = plt.figure(1, [16, 9])

        for k, v in dic.items():
            epochs = range(1, len(v) + 1)
            plt.plot(epochs, v, label=k)

    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel(f"{metricName}", fontsize=20)
    plt.title('Training history', fontsize=20)
    plt.legend()

    # Print image
    name = f'figures/{filename}.png'
    plt.savefig(name, bbox_inches='tight', dpi=300)
    plt.clf()


def plot_auroc(y_test, y_pred, labelDict, filename):
    """
    Generates an auroc plot
    Assumes that y_test and y_pred are in sparse format e.g [.8, .2, .1] or [1, 0, 0]
    Args:
        y_test: reference y values
        y_pred: model predicted y values
        labelDict: dictionary of labels, e.g {"Human": 1, "Unknown": 0}
        filename: filename to write to
    """
    if filename:
        with plt.style.context("seaborn"):
            fig = plt.figure(1, [16, 9])
            for label, i in labelDict.items():
                fpr, tpr, thresholds = roc_curve(y_test[:, i], y_pred[:, i])
                plt.plot(fpr, tpr, label='%s (AUC:%0.5f)' % (label, auc(fpr, tpr)))
            plt.plot(fpr, fpr, 'b-', label='Random Guessing')

        plt.xlabel('False positive rate', fontsize=20)
        plt.ylabel("True positive rate", fontsize=20)
        plt.title('ROC', fontsize=20)
        plt.legend()

        # Print image
        name = f'figures/{filename}.png'
        plt.savefig(name, bbox_inches='tight', dpi=300)
        plt.clf()


def plot_umap(args, adata):
    sc.set_figure_params(dpi_save=500)
    sc.tl.umap(adata, min_dist=1.)
    plt.rcParams['figure.figsize'] = (9, 9)
    for key in adata.obs.columns:
        sc.pl.umap(adata, color=key, save=f'_{args.namespace}_{key}.png')
