from random import shuffle

import numpy as np
from scipy.misc import imresize

from activations import sigmoid, d_sigmoid
from loss import squared_error, d_squared_error
from matrix import dot, multiply, transpose, mean, scalar_mul, matrix_plus
from utils import initialize_bias, initialize_weights, get_split, vectorize_data, to_categorical, flatten, get_batches, \
    batch_errors_mean, to_label
import pickle


class OutputLayer:
    def __init__(self, input_shape=tuple(), weights_range=(-0.2, 0.2), activation="sigmoid"):
        self.weights = initialize_weights(weights_range=weights_range, input_shape=input_shape)
        self.bias = initialize_bias(input_shape=input_shape[0])
        self.error = squared_error
        self.d_error = d_squared_error
        self.activation = sigmoid
        self.d_activation = d_sigmoid
        self.output = None
        self.delta = None

    def propagate(self, x):
        self.input = x
        all_neurons_input = x * len(self.weights)
        layer_net = dot(self.weights, all_neurons_input)
        self.output = [[self.activation(sum(layer_net[i]) + self.bias[0][i])
                        for i in range(len(layer_net))]]
        return self.output

    def back_propagate(self, label):
        slopes = [[self.d_activation(self.output[i][j])
                   for j in range(len(self.output[i]))]
                   for i in range(len(self.output))]
        self.delta = transpose(dot(self.d_error(label, self.output), slopes))
        return multiply(transpose(self.weights), self.delta)


class Dense:
    def __init__(self, input_shape=tuple(), weights_range=(-0.2, 0.2), activation="sigmoid"):
        self.weights = initialize_weights(weights_range=weights_range, input_shape=input_shape)
        self.bias = initialize_bias(input_shape=input_shape[0])
        self.activation = sigmoid
        self.d_activation = d_sigmoid
        self.input = None
        self.output = None
        self.slopes = None
        self.d_output = None
        self.back_output = None

    def propagate(self, x):
        self.input = x
        all_neurons_input = x * len(self.weights)
        layer_net = dot(self.weights, all_neurons_input)
        self.output = [[self.activation(sum(layer_net[i]) + self.bias[0][i])
                        for i in range(len(layer_net))]]
        return self.output

    def back_propagate(self, error):
        slopes = transpose([[self.d_activation(self.output[i][j])
                             for j in range(len(self.output[i]))]
                             for i in range(len(self.output))])
        self.delta = dot(error, slopes)
        return multiply(transpose(self.weights), self.delta)


class Sequential:
    def __init__(self, layers=[], optimizer='SGD', error='d_squared_error'):
        self.layers = layers
        self.optimizer = optimizer
        self.output = None
        self.d_error = d_squared_error
        self.training_errors = []

    def add(self, layer):
        self.layers.append(layer)

    def train(self, x, y, batch_size=4, epochs=100, eta=0.1,
              shuffle_data=True, validation_split=0.1, test_split=0.1,
              validation_data=None):
        labeled_data = list(zip(x, y))
        shuffle(labeled_data)
        train_data, validation_data_, test_data = self.split_data(labeled_data, test_split, validation_split)
        if validation_data is None:
            validation_data = list(zip(*validation_data_))
        for epoch in range(epochs):
            if shuffle_data:
                shuffle(train_data)
            print("Epoch: {}".format(epoch))
            for batch in get_batches(batch_size, train_data):
                self.batch_update(batch, eta)
            self.training_errors.append(self.validation_error(*validation_data))
            validate_result = self.validate(*validation_data)
            print("Validation: {}".format(validate_result))

    def batch_update(self, batch, eta):
        batch_errors = []
        for sample, label in batch:
            self.calculate_activation(sample)
            batch_errors.append(self._calculate_errors(label))
        for i in range(len(batch)):
            batch_errors[i] = batch_errors[i][::-1]
        batch_errors = batch_errors_mean(batch_errors)
        for i in range(len(self.layers)-1, 0, -1):
            weight_update_values = scalar_mul(multiply(transpose(batch_errors[0][i]), self.layers[i].input), eta)
            self.layers[i].weights = matrix_plus(self.layers[i].weights, weight_update_values)
            self.layers[i].bias = matrix_plus(self.layers[i].bias, scalar_mul(batch_errors[0][i], eta))
        return batch_errors[0][-1]

    def _calculate_errors(self, label):
        errors = []
        error = label
        for layer in reversed(self.layers):
            error = layer.back_propagate(error)
            errors.append(transpose(layer.delta))
        return errors

    def calculate_activation(self, x):
        output = x
        for layer in self.layers:
            output = layer.propagate(output)
        self.output = output
        return output

    def predict(self, x):
        max_index, max_value = -1, -1
        output = self.calculate_activation(x)
        for i in range(len(output[0])):
            class_output = output[0][i]
            if max_value < class_output:
                max_index, max_value = i, class_output
        return max_index

    def validate(self, x, y):
        x_size = len(x)
        errors = 0
        for i in range(x_size):
            result = self.predict(x[i])
            errors += int(to_label(y[i]) != result)
        return 1 - (errors / x_size)

    def validation_error(self, xs, ys):
        output = []
        error_function = self.layers[-1].error
        for i in range(len(xs)):
            output.append(error_function(ys[i], self.calculate_activation(xs[i])))
        return transpose(mean(transpose(output)))

    def split_data(self, labeled_data, test_split, validation_split):
        data_test_split = len(labeled_data) * test_split
        data_validation_split = len(labeled_data) * validation_split
        train_split_index = int(len(labeled_data) * 1 - data_validation_split - data_test_split)
        validation_split_index = int(len(labeled_data) * 1 - data_test_split)
        train_data = get_split(labeled_data, 0, train_split_index)
        validation_data = get_split(labeled_data, train_split_index, validation_split_index)
        test_data = get_split(labeled_data, validation_split_index, len(labeled_data))
        return train_data, validation_data, test_data


def main():
    # from keras.datasets import mnist
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train = np.array([255 - imresize(x, (10, 7), interp='bilinear', mode=None) for x in x_train])
    # x_train = x_train / 255
    # y_train = to_categorical(y_train)
    data = np.load("/home/piotr/PWr/SN-L/Neural_Network/visualization_data.npy")
    data = data / 255
    validation_data = data
    # data = np.concatenate((data, x_train), axis=0)
    labels = np.load("/home/piotr/PWr/SN-L/Neural_Network/hot_labels.npy")
    validation_labels = labels
    # labels = np.concatenate((labels, y_train), axis=0)
    model = Sequential()
    model.add(Dense(input_shape=(200, 70)))
    model.add(OutputLayer(input_shape=(10, 200)))

    vectorized_data = vectorize_data(data).tolist()
    model.train(vectorized_data, labels.tolist())#, validation_data=(vectorize_data(validation_data).tolist(),
                                                 #                  validation_labels.tolist()))

if __name__ == "__main__":
    main()
