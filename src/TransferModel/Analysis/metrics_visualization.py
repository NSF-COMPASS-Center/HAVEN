import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


def box_plot(df, output_file_path, x_col, y_col, y_label, random_predictor_val=None):
    plt.clf()
    ax = sns.boxplot(x=df[x_col], y=df[y_col], orient="v")
    ax.set_ylabel(y_label)
    if random_predictor_val is not None:
        ax.axhline(random_predictor_val, color="gray", linestyle="--")
    ax.set_ylim(0, 1)
    plt.savefig(output_file_path)


def binary_precision_recall_curves(df, y_test_col, y_pred_col, itr_col, output_file_path):
    plt.clf()
    itrs = df[itr_col].unique()
    ax = plt.subplot(1, 1, 1)
    for itr in itrs:
        itr_df = df[df[itr_col] == itr]
        PrecisionRecallDisplay.from_predictions(itr_df[y_test_col].values, itr_df[y_pred_col].values, name="Iteration: " + str(itr), ax=ax)
    plt.savefig(output_file_path)


def binary_roc_curves(df, y_test_col, y_pred_col, itr_col, output_file_path):
    plt.clf()
    itrs = df[itr_col].unique()
    ax = plt.subplot(1, 1, 1)
    for itr in itrs:
        itr_df = df[df[itr_col] == itr]
        RocCurveDisplay.from_predictions(itr_df[y_test_col].values, itr_df[y_pred_col].values, name="Iteration: " + str(itr), ax=ax)
    plt.savefig(output_file_path)