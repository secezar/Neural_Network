from math import exp

def sigmoid(x):
    return (1 /(1 + exp(-round(x, 6))))


def d_sigmoid(x):
    return x * (1 - x)


def softmax(xs):
    x_exp = [exp(x) for x in xs]
    x_sum = sum(x_exp)
    return [x_exp[i] / x_sum for i in range(len(x_exp))]


