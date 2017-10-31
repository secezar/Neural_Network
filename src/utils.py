from random import random, uniform


def initialize_weights(weights_range=(-1, 1), input_len=0):
    return [uniform(*weights_range) for _ in range(input_len)]


def initialize_bias():
    return random()
