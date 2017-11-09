from random import random, uniform

import numpy as np


def initialize_weights(weights_range=(-1, 1), input_shape=(0,0)):
    return [[uniform(*weights_range) for _ in range(input_shape[1])] for _ in range(input_shape[0])]


def initialize_bias():
    return random()


def get_batches(batch_size, train_data):
    return [train_data[i:i + batch_size] for i in range(0, len(train_data), batch_size)]


def get_split(data, start, stop):
    return data[start:stop]


def to_categorical(y, num_classes=None):
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical


def vectorize_data(matrix):
    return np.reshape(matrix, (matrix.shape[0], 1, matrix.shape[1] * matrix.shape[2]))


flatten = lambda l: [item for sublist in l for item in sublist]