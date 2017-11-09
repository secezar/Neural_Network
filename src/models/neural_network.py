import os
from random import shuffle, random, choice

import cv2
import numpy as np

from activations import sigmoid, d_sigmoid
from matrix import dot, multiply, transpose, row_sum, plus
from utils import initialize_bias, initialize_weights, get_split, vectorize_data, to_categorical, flatten, get_batches


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
    def __init__(self, input_shape=tuple(), weights_range=(-1, 1), activation="sigmoid"):
        self.weights = initialize_weights(weights_range=weights_range, input_shape=input_shape)
        self.bias = initialize_bias()
        self.activation = sigmoid
        self.d_activation = d_sigmoid
        self.output = None
        self.d_output = None
        self.back_output = None

    def propagate(self, x):
        all_neurons_input = [x] * len(self.weights)
        layer_net = dot(self.weights, all_neurons_input)
        self.output = []
        for i in range(len(layer_net)):
            net = sum(layer_net[i]) + self.bias
            activation = self.activation(net)
            self.output.append(activation)
        return self.output

    def back_propagate(self, error):
        self.d_output = []
        for i in range(len(self.output)):
            neuron_activation = self.output[i]
            output_slope = self.d_activation(neuron_activation)
            delta_layer = error[i] * output_slope
            self.d_output.append(delta_layer)
        self.back_output = dot([self.d_output], transpose(self.weights)) #d_output intem
        return [self.back_output]


class Dense:
    def __init__(self, input_shape=tuple(), weights_range=(-1, 1), activation="sigmoid"):
        self.weights = initialize_weights(weights_range=weights_range, input_shape=input_shape)
        self.bias = initialize_bias()
        self.activation = sigmoid
        self.d_activation = d_sigmoid
        self.output = None
        self.d_output = None
        self.back_output = None

    def propagate(self, x):
        all_neurons_input = [x] * len(self.weights)
        layer_net = dot(self.weights, all_neurons_input)
        self.output = []
        for i in range(len(layer_net)):
            net = sum(layer_net[i]) + self.bias
            activation = self.activation(net)
            self.output.append(activation)
        return self.output

    def back_propagate(self, error):
        self.d_output = []
        for i in range(len(self.output)):
            neuron_activation = self.output[i]
            output_slope = self.d_activation(neuron_activation)
            delta_layer = error[i] * output_slope
            self.d_output.append(delta_layer)
        self.back_output = dot([self.d_output], transpose(self.weights)) #d_output intem
        return self.back_output


class Sequential:
    def __init__(self, layers=[], optimizer='SGD'):
        self.layers = layers
        self.optimizer = optimizer

    def add(self, layer):
        self.layers.append(layer)

    def train(self, x, y, batch_size=4, epochs=50, eta=0.001, shuffle_data=True, validation_split=0., test_split=0.):
        labeled_data = list(zip(x, y))
        shuffle(labeled_data)
        train_data, validation_data, test_data = self.split_data(labeled_data, test_split, validation_split)
        for epoch in range(epochs):
            if shuffle_data:
                shuffle(train_data)
            for batch in get_batches(batch_size, train_data):
                self.batch_update(batch, eta)

    def batch_update(self, batch, eta):
        batch_errors = [self._calculate_errors(sample, label) for sample, label in batch]
        for layer, error in zip(reversed(self.layers), transpose(batch_errors)):
            layer.weights = layer.weights + eta * dot(transpose(layer.output), error)
            layer.bias = layer.bias + sum(error) * eta

    def _calculate_errors(self, sample, label):
        errors = []
        error = self.error(label, self.calculate_activation(sample))
        print(error)
        errors.append(error)
        for layer in reversed(self.layers):
            error = layer.back_propagate(error)
            errors.append(errors)
        return errors

    def calculate_activation(self, x):
        output = x
        for layer in self.layers:
            output = layer.propagate(output)
        return output

    def predict(self, x):
        max_index, max_value = -1, -1
        output = self.calculate_activation(x)
        for i in range(len(output)):
            class_output = output[i]
            if max_value < class_output:
                max_index, max_value = i, class_output
        return max_index

    def error(self, label, activations):
        return [label[i] - activations[i] for i in range(len(activations))]

    def validate(self, x, y):
        x_size = len(x)
        errors = 0
        for i in range(x_size):
            errors += self.error(y[i], self.predict(x))
        return 1 - (errors / x_size)

    def split_data(self, labeled_data, test_split, validation_split):
        train_split_index = int(len(labeled_data) * 1 - validation_split - test_split)
        validation_split_index = int(len(labeled_data) * 1 - test_split)
        train_data = get_split(labeled_data, 0, train_split_index)
        validation_data = get_split(labeled_data, train_split_index, validation_split_index)
        test_data = get_split(labeled_data, validation_split_index, len(labeled_data))
        return train_data, validation_data, test_data

def main():
    data = np.load("C:\\Users\\Jola\\Desktop\\PROJEKTY\\SN\\Neural_Network\\visualization_data.npy")
    print(data.shape)
    data = data/255
    labels = np.load("C:\\Users\\Jola\\Desktop\\PROJEKTY\\SN\\Neural_Network\\hot_labels.npy")
    model = Sequential()
    model.add(Dense(input_shape=(70, 70)))
    model.add(Dense(input_shape=(70, 30)))
    model.add(Dense(input_shape=(30, 10)))
    model.add(Dense(input_shape=(10, 1)))
    model.train(flatten(vectorize_data(data).tolist()), labels.tolist())

if __name__ == "__main__":
    main()
