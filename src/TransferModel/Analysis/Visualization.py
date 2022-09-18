import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from sklearn.metrics import roc_curve, auc

import scanpy as sc


def plot_train_test_metric(trainMetric, testMetric, filename, metricName="Cross Entropy", optimalFlag="min",
                           ylim=[0, 1]):
    """
    Used for plotting train and test metrics indexed by epochs to files
    Args:
        trainMetric: Array of test metric values indexed by epoch
        testMetric: Array of train metric values indexed by epoch
        filename: path and filename to write to
        metricName: y-axis name
        optimalFlag: plots a horizontal line depending on the optimal epoch choices are: ["min", "max", None]
        ylim: plot [ymin, yMax] limits
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
    ax = plt.gca()
    ax.set_ylim(ylim)

    # Print image
    name = f'{filename}.png'
    plt.savefig(name, bbox_inches='tight', dpi=300)
    plt.clf()


def plot_metrics(dic, filename, metricName="AUROC", ylim=[0, 1]):
    """
    Plots metrics in figures/filename.png
    Args:
        dic: Dictionary of label:[data1, data2, ...] pairs
        filename: path and filename to write to
        metricName: y axis label
        ylim: plot [ymin, yMax] limits
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
    ax = plt.gca()
    ax.set_ylim(ylim)

    # Print image
    name = f'{filename}.png'
    plt.savefig(name, bbox_inches='tight', dpi=300)
    plt.clf()


def plot_auroc(y_test, y_pred, labelDict, filename, ylim=[0, 1]):
    """
    Generates an auroc plot
    Assumes that y_test and y_pred are in sparse format e.g [.8, .2, .1] or [1, 0, 0]
    Args:
        y_test: reference y values
        y_pred: model predicted y values
        labelDict: dictionary of labels, e.g {"Human": 1, "Unknown": 0}
        filename: path and filename to write to
        ylim: plot [ymin, yMax] limits
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
        ax = plt.gca()
        ax.set_ylim(ylim)

        # Print image
        name = f'{filename}.png'
        plt.savefig(name, bbox_inches='tight', dpi=300)
        plt.clf()


def plot_umap(args, adata, figdir):
    sc.settings.figdir = figdir
    sc.tl.umap(adata, min_dist=1.)
    sc.set_figure_params(dpi_save=500)
    plt.rcParams['figure.figsize'] = (9, 9)
    for key in adata.obs.columns:
        sc.pl.umap(adata, color=key, save=f'_{args.namespace}_{key}.png')

def plot_umap3d(args, adata, figdir):
    sc.settings.figdir = figdir
    sc.tl.umap(adata, min_dist=1., n_components=3)
    sc.set_figure_params(dpi_save=150)
    plt.rcParams['figure.figsize'] = (9, 9)
    fps = 30
    interval = 50 #ms inbetween frames in realtime, essentially speed of the animation
    for key in adata.obs.columns:
        fig = sc.pl.umap(adata, color=key, return_fig=True, projection="3d")
        fig, ax = plt.gcf(), plt.gca()
        plt.tight_layout()
        rot_animation = animation.FuncAnimation(fig, lambda angle: ax.view_init(azim=angle), frames=np.arange(0, 361, .5), interval=interval)
        rot_animation.save(f'{figdir}/umap_{args.namespace}_{key}_3d.gif', fps=fps, writer='pillow', dpi=150)
