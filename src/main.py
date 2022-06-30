from itertools import cycle, islice

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.algorithm import transform
from src.load_data import load_orl, load_sklearn_datasets
from src.utils import dimensions_reduction


def main():
    data, labels = load_orl()
    data = dimensions_reduction(data)
    y_pred = transform(data, 10)
    print(y_pred)


def main_two():
    datasets = load_sklearn_datasets()
    # ============
    # Set up cluster parameters
    # ============
    plt.figure(figsize=(5, 13))
    plt.subplots_adjust(
        left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01
    )

    plot_num = 1

    for i_dataset, (dataset, algo_params) in enumerate(datasets):
        X, y = dataset

        # normalize dataset for easier parameter selection
        X = StandardScaler().fit_transform(X)
        y_pred = transform(X, algo_params["n_clusters"])
        plt.subplot(len(datasets), 1, plot_num)

        colors = np.array(
            list(islice(cycle([
                "#377eb8",
                "#ff7f00",
                "#4daf4a",
                "#f781bf",
                "#a65628",
                "#984ea3",
                "#999999",
                "#e41a1c",
                "#dede00",
            ]), int(max(y_pred) + 1), )))
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plot_num += 1

    plt.show()


if __name__ == '__main__':
    main_two()
