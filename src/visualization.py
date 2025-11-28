import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8")
sns.set()


def plot_class_distribution(y: np.ndarray, ax=None, labels=None, title="Class distribution"):
    """
    Pie chart phân bố lớp (y = 0/1).
    """
    if labels is None:
        labels = ["Class 0", "Class 1"]

    n0 = np.sum(y == 0)
    n1 = np.sum(y == 1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    ax.pie([n0, n1], labels=labels, autopct="%1.2f%%", startangle=90)
    ax.set_title(title)
    return ax


def plot_histogram(x: np.ndarray, bins=30, xlabel="", ylabel="Count", title="", logy=False):
    """
    Histogram đơn giản cho một biến.
    """
    plt.figure(figsize=(6, 4))
    plt.hist(x, bins=bins, edgecolor="black")
    if logy:
        plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_histogram_by_class(x: np.ndarray, y: np.ndarray,
                            bins=30,
                            labels=("Class 0", "Class 1"),
                            xlabel="",
                            ylabel="Density",
                            title=""):
    """
    Histogram so sánh phân phối x giữa 2 lớp (0 và 1).
    """
    x0 = x[y == 0]
    x1 = x[y == 1]

    plt.figure(figsize=(6, 4))
    plt.hist(x0, bins=bins, alpha=0.5, label=labels[0], density=True)
    plt.hist(x1, bins=bins, alpha=0.7, label=labels[1], density=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm: np.ndarray,
                          labels=("Pred 0", "Pred 1"),
                          true_labels=("True 0", "True 1"),
                          title="Confusion Matrix"):
    """
    Vẽ confusion matrix (2x2) dùng seaborn.heatmap

    cm: np.array([[tn, fp],
                  [fn, tp]])
    """
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cbar=False,
                xticklabels=labels,
                yticklabels=true_labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()
