import os
from random import shuffle, random

import cv2
import numpy as np

from activations import sigmoid, d_sigmoid
from matrix import dot, multiply, transpose
from utils import initialize_bias, initialize_weights


class Data:
    def __init__(self):
        self.data = []
        self.labels = []

    def load_dir_dataset(self, dir_path):
        for path in os.listdir(dir_path):
            self.data.append(self.load(os.path.join(dir_path, path)))
            self.labels.append(int(path[0]))
        return np.array(self.data), np.array(self.labels)

    def load(self, path):
        return cv2.imread(path)

    def load_x_dataset(self, array_path):
        assert array_path[-4:] == ".npy", "{}".format(array_path[-4:])
        self.data = np.load(array_path)

    def load_y_dataset(self, array_path):
        assert array_path[-4:] == ".npy", "{}".format(array_path[-4:])
        self.labels = np.load(array_path)

    def save(self):
        np.save("data", np.array(self.data))
        np.save("labels", np.array(self.labels))

    def show_sample(self, index):
        print(self.data[index].shape)
        cv2.imshow("Sample", self.data[index])
        cv2.waitKey()


class OutputLayer:
    def __init__(self, neurons, weights_range=(-1, 1), activation="sigmoid"):
        self.weights = initialize_weights(weights_range=weights_range, input_len=neurons)
        self.bias = initialize_bias()
        self.activation = sigmoid
        self.d_activation = d_sigmoid
        self.output = None
        self.back_output = None

    def propagate(self, x):
        self.output = [self.activation(sum(dot(self.weights[i], x)) + self.bias) for i in range(len(self.weights))]
        return self.output

    def back_propagate(self, error):
        self.d_output = [dot(error, self.d_activation(self.output[i])) for i in range(len(self.output))]
        self.back_output = multiply(self.d_output, transpose(self.weights))
        return self.back_output


class HiddenLayer:
    def __init__(self, neurons, weights_range=(-1, 1), activation="sigmoid", optimizer="SGD"):
        self.weights = initialize_weights(weights_range=weights_range, input_len=neurons)
        self.bias = initialize_bias()
        self.activation = sigmoid
        self.d_activation = d_sigmoid
        self.output = None
        self.back_output = None

    def propagate(self, x):
        self.output = self.activation(sum(dot(self.weights, x)) + self.bias)
        return self.output

    def back_propagate(self, error):
        self.d_output = dot(error, self.d_activation(self.output))
        self.back_output = dot(self.d_output, transpose(self.weights))
        return self.back_output


class Sequential:
    def __init__(self, layers=[], optimizer='SGD'):
        self.layers = layers
        self.optimizer = optimizer

    def add(self, layer):
        self.layers.append(layer)

    def train(self, x, y, batch_size=4, epochs=50, learning_rate=0.001, shuffle_data=True):
        labeled_data = list(zip(x, y))
        for epoch in range(epochs):
            if shuffle_data:
                shuffle(labeled_data)
            batch = [random.choice(labeled_data) for _ in range(batch_size)]
            self.batch_update(batch, learning_rate)

    def batch_update(self, batch, eta):
        nabla_b = []
        nabla_w = []
        for sample, label in batch:
            error = self.error(label, self.calculate_activation(sample))
            for layer in reversed(self.layers):
                delta_nabla_b, delta_nabla_w = layer.back_propagation(error)
                nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
                layer.bias = [b - (eta / len(batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def predict(self, x):
        return max(self.calculate_activation(x))

    def calculate_activation(self, x):
        output = x
        for layer in self.layers:
            output = layer.propagate(x)
        return output

    def error(self, label, activations):
        return [label - activation for activation in activations]

    def validate(self, x, y):
        x_size = len(x)
        errors = 0
        for i in range(x_size):
            errors += self.error(y[i], self.predict(x))
        return 1 - (errors / x_size)


def main():


if __name__ == "__main__":
    main()
