from random import shuffle

import numpy as np

from activations import sigmoid, d_sigmoid
from loss import squared_error
from matrix import dot, multiply, transpose, mean, scalar_mul, matrix_plus
from utils import initialize_bias, initialize_weights, get_split, vectorize_data, to_categorical, flatten, get_batches, \
    batch_errors_mean, to_label


class OutputLayer:
    def __init__(self, input_shape=tuple(), weights_range=(-1, 1), activation="sigmoid"):
        self.weights = initialize_weights(weights_range=weights_range, input_shape=input_shape)
        self.bias = initialize_bias(input_shape=input_shape[0])
        self.error = squared_error
        self.activation = sigmoid
        self.d_activation = d_sigmoid
        self.output = None
        self.slopes = None

    def propagate(self, x):
        self.output = x
        return x

    def back_propagate(self, label):
        self.slopes = [[self.d_activation(self.output[i][j])
                        for j in range(len(self.output[i]))]
                       for i in range(len(self.output))]
        return dot(self.error(label, self.output), self.slopes)


class Dense:
    def __init__(self, input_shape=tuple(), weights_range=(-1, 1), activation="sigmoid"):
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
        self.slopes = [[self.d_activation(self.output[i][j])
                       for j in range(len(self.output[i]))]
                       for i in range(len(self.output))]
        self.d_output = dot(error, self.slopes)
        self.back_output = multiply(self.d_output, self.weights)
        return self.back_output


class Sequential:
    def __init__(self, layers=[], optimizer='SGD', error='squared_error'):
        self.layers = layers
        self.optimizer = optimizer
        self.output = None
        self.error = squared_error

    def add(self, layer):
        self.layers.append(layer)

    def train(self, x, y, batch_size=1, epochs=50, eta=0.1, shuffle_data=True, validation_split=0.1, test_split=0.1):
        labeled_data = list(zip(x, y))
        shuffle(labeled_data)
        train_data, validation_data, test_data = self.split_data(labeled_data, test_split, validation_split)
        validation_data = list(zip(*validation_data))
        for epoch in range(epochs):
            if shuffle_data:
                shuffle(train_data)
            print("Epoch: {}".format(epoch))
            for batch in get_batches(batch_size, train_data):
                self.batch_update(batch, eta)
            print(self.layers[-1].weights)
            print("Validation: {}".format(self.validate(*validation_data)))

    def batch_update(self, batch, eta):
        batch_errors = []
        for sample, label in batch:
            self.calculate_activation(sample)
            batch_errors.append(self._calculate_errors(label))
        batch_errors = batch_errors_mean(batch_errors)
        for i in range(len(self.layers)-1, 0, -1):
            weight_update_values = scalar_mul(multiply(transpose(self.layers[i].input), [batch_errors[i]]), -eta)
            self.layers[i-1].weights = matrix_plus(self.layers[i-1].weights, weight_update_values)
            self.layers[i-1].bias = matrix_plus(self.layers[i-1].bias, scalar_mul([batch_errors[i-1]], -eta))

    def _calculate_errors(self, label):
        errors = []
        error = self.error(label, self.output)
        for layer in reversed(self.layers):
            error = layer.back_propagate(error)
            errors.append(error)
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
    data = np.load("/home/piotr/PWr/SN-L/Neural_Network/visualization_data.npy")
    print(data.shape)
    data = data/255
    labels = np.load("/home/piotr/PWr/SN-L/Neural_Network/hot_labels.npy")
    model = Sequential()
    model.add(Dense(input_shape=(100, 70)))
    model.add(Dense(input_shape=(10, 100)))
    vectorized_data = vectorize_data(data).tolist()
    model.train(vectorized_data, labels.tolist())

if __name__ == "__main__":
    main()
