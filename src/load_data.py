import os

import cv2 as cv
import numpy as np
from sklearn import datasets

# Save image in set directory
# Read RGB image

orl_path = "dataset/ORL/"


def load_orl():
    files = os.listdir(orl_path)
    labels = np.zeros(400)
    data = np.zeros((400, 70 * 80))
    for i, file in enumerate(files):
        label = int(file.split(".")[0].split("_")[1])
        file_path = orl_path + file
        img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
        img = img.flatten()
        labels[i] = label
        data[i] = img
    return data, labels


def load_sklearn_datasets():
    n_samples = 500
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
    blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None

    # Anisotropicly distributed data
    random_state = 170
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = datasets.make_blobs(
        n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
    )

    temp_datasets = [
        (
            noisy_circles,
            {
                "n_clusters": 2,
            },
        ),
        (
            noisy_moons,
            {
                "n_clusters": 2,
            },
        ),
        (
            varied,
            {
                "n_clusters": 3,
            },
        ),
        (
            aniso,
            {
                "n_clusters": 3,
            },
        ),
        (
            blobs,
            {
                "n_clusters": 3,
            }
        ),
        (
            no_structure,
            {
                "n_clusters": 2,
            }
        ),
    ]

    return temp_datasets


def main():
    data, labels = load_orl()
    print(data.shape)
    print(labels.shape)


if __name__ == '__main__':
    main()
