import math

import numpy as np
from sklearn.metrics import pairwise_distances


def calculate_distance_matrix(data):
    distance_matrix = pairwise_distances(data)
    return distance_matrix


def create_distance_matrix_current(distance_matrix, k):
    distance_matrix_current = np.zeros(distance_matrix.shape)
    constant = 1 / math.pow(k + 1, 2)
    k_i = np.empty((distance_matrix.shape[0], k))
    k_i[:, 0] = np.nan
    for i in range(distance_matrix.shape[0]):
        k_i = find_k_nearest_neighbor(distance_matrix, i, k, k_i)
        for j in range(i, distance_matrix.shape[1]):
            k_i = find_k_nearest_neighbor(distance_matrix, j, k, k_i)

            temp_distance = 0
            for i_index in k_i[i]:
                for j_index in k_i[j]:
                    temp_distance += distance_matrix[int(i_index), int(j_index)]
            for i_index in k_i[i]:
                temp_distance += distance_matrix[int(i_index), j]
            for j_index in k_i[j]:
                temp_distance += distance_matrix[i, int(j_index)]

            temp_distance *= constant
            distance_matrix_current[i, j] = temp_distance
            distance_matrix_current[j, i] = temp_distance
    return distance_matrix_current


def find_k_nearest_neighbor(distance_matrix, i, k, k_i):
    if np.isnan(k_i[i, 0]):
        r_i = distance_matrix[i].argsort()[1:k + 1]
        k_i[i] = r_i
    return k_i


def find_key_points(distance_matrix, c_target):
    # find first key point
    mean_distance = np.divide(distance_matrix.sum(axis=1), distance_matrix.shape[0] - 1)
    key_points = [mean_distance.argmin()]

    for i in range(1, c_target):
        temp_max = -1
        temp_max_args = -1
        for j in range(distance_matrix.shape[0]):
            if j in key_points:
                continue
            temp_min = np.inf
            for k in range(len(key_points)):
                temp_distance = distance_matrix[j, key_points[k]]
                if temp_distance < temp_min:
                    temp_min = temp_distance
            if temp_min > temp_max:
                temp_max = temp_min
                temp_max_args = j
        key_points.append(temp_max_args)
    return np.array(key_points)


def update_labels(output_label, distance_matrix_current, s_current):
    output_label = output_label.copy()
    new_s_current = np.concatenate([np.array(s_current).reshape(-1, 1), np.arange(0, len(s_current)).reshape(-1, 1)],
                                   axis=1)
    for i in range(output_label.shape[0]):
        label = output_label[i]
        if i in s_current:
            output_label[i] = new_s_current[np.where(s_current == i)[0][0]][1]
        else:
            output_label[i] = new_s_current[distance_matrix_current[label, s_current].argmin(), 1]
    return output_label


def get_cluster_members(output_label, index, distance_matrix, k, k_i):
    p_i = np.where(output_label == index)[0]
    for i in p_i:
        k_i = find_k_nearest_neighbor(distance_matrix, i, k, k_i)
    p_i = np.unique(np.append(p_i, np.concatenate([k_i[i] for i in p_i]))).astype(np.int32)
    return p_i, k_i


def create_new_distance_matrix_current(output_label, distance_matrix, k):
    k_i = np.empty((distance_matrix.shape[0], k))
    k_i[:, 0] = np.nan

    unique_labels = np.unique(output_label)
    current_cluster_count = unique_labels.shape[0]
    new_distance_matrix = np.zeros((current_cluster_count, current_cluster_count))

    for i in range(current_cluster_count):
        label_i = unique_labels[i]
        p_i, k_i = get_cluster_members(output_label, label_i, distance_matrix, k, k_i)
        for j in range(i, current_cluster_count):
            label_j = unique_labels[j]
            p_j, k_i = get_cluster_members(output_label, label_j, distance_matrix, k, k_i)
            constant = 1 / (p_i.shape[0] * p_j.shape[0])
            temp_distance = 0
            for x in p_i:
                temp_distance += distance_matrix[x, p_j].sum()
            new_distance_matrix[i, j] = constant * temp_distance

    return new_distance_matrix


def transform(data, c_target, k=5, g=5):
    n = data.shape[0]
    output_label = np.arange(0, data.shape[0])
    c_current = math.floor(n / g)
    distance_matrix = calculate_distance_matrix(data)
    distance_matrix_current = create_distance_matrix_current(distance_matrix, k)
    while c_current > c_target:
        s_current = find_key_points(distance_matrix_current, c_current)
        output_label = update_labels(output_label, distance_matrix_current, s_current)
        distance_matrix_current = create_new_distance_matrix_current(output_label, distance_matrix, k)
        c_current = math.floor(c_current / g)
    s_final = find_key_points(distance_matrix_current, c_target)
    output_label = update_labels(output_label, distance_matrix_current, s_final)
    return output_label
