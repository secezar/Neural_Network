from random import shuffle

import cv2

from activations import sigmoid
from utils import initialize_bias, initialize_weights

class Data:
    def __init__(self):
        self.data = []

    def load(self, path):
        self.data = cv2.imread(path)

    def show_sample(self, image_array):
        cv2.imshow("Sample", image_array)


class Layer:
    def __init__(self, neurons, weights_range=(-1, 1)):
        self.weights = initialize_weights(weights_range=weights_range, input_len=neurons)
        self.bias = initialize_bias()


class Activation:
    def __init__(self, name="sigmoid"):
        self.name = name


class Sequential():
    def __init__(self, layers, optimizer):
        self.layers = []
        self.optimizer = optimizer

    def add(self, layer):
        self.layers.append(layer)

    def train(self, x, y, batch_size=4, epochs=50, learning_rate=0.001, shuffle_data=True):
        labeled_data = list(zip(x, y))
        for epoch in range(epochs):
            if shuffle_data:
                shuffle(labeled_data)
            epoch_loss = 0
            for sample, label in labeled_data:
                error = self.error(sample, label)
                delta = learning_rate * error

                for i in range(len(self.weights)):
                    self.weights[i] += delta * sample[i]
                self.bias += delta

    def predict(self, x):
        output = x
        for layer in self.layers:
            output = layer(x)

        return max(output)

    def error(self, label, predicted):
        return int(label == predicted)

    def validate(self, x, y):
        x_size = len(x)
        errors = 0
        for i in range(x_size):
            errors += self.error(y[i], self.predict(x))
        return 1 - (errors / x_size)

