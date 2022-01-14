import matplotlib.pyplot as plt
import numpy as np
import config


def plot_sparsity(df_unit, df_weight):
    """ Plots the accuracies for each k value of the pruner

    Args:
        df_unit ([Object]): [Type of Pruner]
        df_weight ([Object]): [Type ofPruner]
    """
    plt.figure()
    plt.title("Accuracy vs Sparsity")
    plt.plot(df_unit["sparsity"], df_unit["accuracy"], label="Unit-pruning")
    plt.plot(df_weight["sparsity"], df_weight["accuracy"], label="Weight-pruning")
    plt.xlabel("Sparsity (as fraction)")
    plt.ylabel("% Accuracy")
    plt.legend()
    plt.savefig(config.PLOT_PATH)
    plt.show()


def l2(array):
    # L2 regularization
    return np.sqrt(np.sum([i ** 2 for i in array]))
