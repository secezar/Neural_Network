from math import exp

import numpy


def sigmoid(x):
    return (1 /(1 + exp(-x)))


def d_sigmoid(x):
    return x * (1 - x)


def softmax(x):
    return (numpy.exp(x) / numpy.exp(x).sum()).to_list()


