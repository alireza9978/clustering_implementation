import os

import cv2 as cv
import numpy as np

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


def main():
    data, labels = load_orl()
    print(data.shape)
    print(labels.shape)


if __name__ == '__main__':
    main()
