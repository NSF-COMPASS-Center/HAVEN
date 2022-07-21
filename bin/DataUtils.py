#!/usr/bin/env python
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score

class DataUtils:
    @staticmethod
    def split_seqs(seqs, percentTrain=.8, seed=1):
        x_train, x_test, y_train, y_test = train_test_split(list(seqs.keys()), list(seqs.values()), train_size=percentTrain, random_state=seed)

        train_seqs = {k:v for k,v in zip(x_train, y_train)}
        test_seqs = {k:v for k,v in zip(x_test, y_test)}

        return train_seqs, test_seqs



    @staticmethod
    def plot_metric(trainCE, testCE, filename, metricName="Cross Entropy", optimalFlag="min"):
        assert filename
        with plt.style.context("seaborn"):
            fig = plt.figure(1, [16, 9])
            epochs = range(1,len(trainCE)+1)
            if trainCE: 
                plt.plot(epochs, trainCE, label=f"Training {metricName}")
                optimal = min(trainCE) if optimalFlag == 'min' else max(trainCE)
                plt.hlines(y=optimal ,xmin=0,xmax=len(trainCE)+1, label=f"Training set optimum {metricName}: {optimal}",  linestyles='-.', lw=1)
            if testCE:
                plt.plot(epochs, testCE, label=f"Testing {metricName}")
                optimal = min(testCE) if optimalFlag == "min" else max(testCE)
                plt.hlines(y=optimal ,xmin=0,xmax=len(testCE)+1, label=f"Testing set optimum {metricName}: {optimal}", linestyles='--', lw=1)
            
        plt.xlabel('Epoch', fontsize=20)
        plt.ylabel(f"{metricName}", fontsize=20)
        plt.title('Training history',fontsize=20)
        plt.legend()

        # Print image
        name = f'figures/{filename}.png'
        plt.savefig(name, bbox_inches='tight', dpi=300)
        plt.clf()

    # Expects y_test, y_pred in softmax format (numerical, not sparse encoding)
    # If filename not provided, will not write to file
    @staticmethod
    def plot_auroc(y_test, y_pred, labelDict, filename, average="macro"):
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)

        print(y_test)
        print(y_pred)
        print(labelDict)

        if filename:
            with plt.style.context("seaborn"):
                fig = plt.figure(1, [16, 9])
                for label, i in labelDict.items():
                    fpr, tpr, thresholds = roc_curve(y_test[:,i-1].astype(int), y_pred[:,i-1])
                    plt.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (label, auc(fpr, tpr)))
                plt.plot(fpr, fpr, 'b-', label = 'Random Guessing')

            plt.xlabel('False positive rate', fontsize=20)
            plt.ylabel("True positive rate", fontsize=20)
            plt.title('ROC',fontsize=20)
            plt.legend()

            # Print image
            name = f'figures/{filename}.png'
            plt.savefig(name, bbox_inches='tight', dpi=300)
            plt.clf()

        return roc_auc_score(y_test, y_pred, average=average)









