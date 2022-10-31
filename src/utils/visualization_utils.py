import matplotlib.pyplot as plt
import seaborn as sns


def box_plot(df, values_col, output_file_path, baseline=None):
    plt.clf()
    sns.set_theme()

    ax = sns.boxplot(x=df[values_col], orient="v")

    if baseline is not None:
        ax.axhline(baseline, color="gray", linestyle="--")
    ax.set_xlim(0, 1)
    # plt.tight_layout()
    plt.savefig(output_file_path)