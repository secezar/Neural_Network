from random import random, uniform

import cv2
import numpy as np
import pickle

from  matrix import mean, transpose


def import_model(path):
    return pickle.load(open(path, 'rb'))


def export_model(model, path):
    pickle.dump(model, open(path, 'wb'))


def initialize_weights(weights_range=(-1, 1), input_shape=(0,0)):
    return [[uniform(*weights_range) for _ in range(input_shape[1])] for _ in range(input_shape[0])]


def initialize_bias(input_shape=0):
    return [[random() for _ in range(input_shape)]]


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


def to_label(y):
    for i in range(len(y)):
        if y[i] == 1:
            return i


def vectorize_data(matrix):
    return np.reshape(matrix, (matrix.shape[0], 1, matrix.shape[1] * matrix.shape[2]))


def batch_errors_mean(batch):
    layers = [[] for _ in batch[0]]
    for sample in range(len(batch)):
        for i in range(len(batch[sample])):
            layers[i].append(batch[sample][i][0])
    return [[[[(mean(elem)) for elem in zip(*layer)]] for layer in layers]]


def get_one_channel(matrix):
    a = matrix.shape
    b = np.squeeze(matrix, axis=3).shape
    for elem in matrix:
        # gray_images = cv2.cvtColor(elem, cv2.COLOR_BGR2GRAY)
        print(elem)

flatten = lambda l: [item for sublist in l for item in sublist]
