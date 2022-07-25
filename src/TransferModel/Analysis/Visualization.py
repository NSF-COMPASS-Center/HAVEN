import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelBinarizer


def plot_metric(trainCE, testCE, filename, metricName="Cross Entropy", optimalFlag="min"):
    assert filename
    with plt.style.context("seaborn"):
        fig = plt.figure(1, [16, 9])
        epochs = range(1, len(trainCE) + 1)
        if trainCE:
            plt.plot(epochs, trainCE, label=f"Training {metricName}")
            optimal = min(trainCE) if optimalFlag == 'min' else max(trainCE)
            plt.hlines(y=optimal, xmin=0, xmax=len(trainCE) + 1,
                       label=f"Training set optimum {metricName}: {optimal}", linestyles='-.', lw=1)
        if testCE:
            plt.plot(epochs, testCE, label=f"Testing {metricName}")
            optimal = min(testCE) if optimalFlag == "min" else max(testCE)
            plt.hlines(y=optimal, xmin=0, xmax=len(testCE) + 1,
                       label=f"Testing set optimum {metricName}: {optimal}", linestyles='--', lw=1)

    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel(f"{metricName}", fontsize=20)
    plt.title('Training history', fontsize=20)
    plt.legend()

    # Print image
    name = f'figures/{filename}.png'
    plt.savefig(name, bbox_inches='tight', dpi=300)
    plt.clf()

# dic : dictioanry of name:[data] pairs
def plot_metrics(dic, filename, metricName="AUROC"):

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


# Expects y_test, y_pred in softmax format (numerical, not sparse encoding)
# If filename not provided, will not write to file
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
                fpr, tpr, thresholds = roc_curve(y_test[:, i].astype(int), y_pred[:, i])
                plt.plot(fpr, tpr, label='%s (AUC:%0.2f)' % (label, auc(fpr, tpr)))
            plt.plot(fpr, fpr, 'b-', label='Random Guessing')

        plt.xlabel('False positive rate', fontsize=20)
        plt.ylabel("True positive rate", fontsize=20)
        plt.title('ROC', fontsize=20)
        plt.legend()

        # Print image
        name = f'figures/{filename}.png'
        plt.savefig(name, bbox_inches='tight', dpi=300)
        plt.clf()

    return roc_auc_score(y_test, y_pred, average=average)
