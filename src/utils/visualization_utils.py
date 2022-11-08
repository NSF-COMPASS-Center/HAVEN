import matplotlib.pyplot as plt
import seaborn as sns


def box_plot(df, values_col, output_file_path, baseline=None):
    plt.clf()
    sns.set_theme()

    ax = sns.boxplot(y=df[values_col])

    if baseline is not None:
        ax.axvline(baseline, color="gray", linestyle="--")
    ax.set_ylim(0, 1)
    # plt.tight_layout()
    plt.savefig(output_file_path)


def class_distribution_plot(df, output_file_path):
    plt.clf()
    sns.set_theme()

    ax = sns.barplot(data=df, x="label", y="label_count", hue="group")

    plt.savefig(output_file_path)