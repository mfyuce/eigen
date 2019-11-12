import numpy as np


def covert_to_float(data):
    try:
        return float(data)
    except:
        return data


def read_matrix_file(file_name):
    with open(file_name, 'r') as f:
        l = [[covert_to_float(item) for item in line.strip().split("\t")] for line in f]
        return l


def read_matrix(file_name):
    l_with_text = read_matrix_file(file_name)
    l = []
    for line in l_with_text:
        i = []
        for item in line:
            if not type(item) is str:
                i.append(item)
        if i:
            l.append(i)
    np_arr = None
    if l:
        np_arr = np.array(l)
    return np_arr, l_with_text, l


def generate_importance_correlation_matrix(np_arr):
    size = len(np_arr)
    f = lambda i, j: np_arr[i][0] / np_arr[j][0] if np_arr[j][0] != 0 else np_arr[j][0]
    return np.fromfunction(np.vectorize(f), (size, size), dtype=int)
