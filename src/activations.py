from numpy import exp


def sigmoid(x):
    return 1 / 1 + exp(-x)

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

