from matrix import scalar_power, scalar_mul


def d_squared_error(label, output):
    return [[label[i] - output[0][i] for i in range(len(output[0]))]]


def squared_error(label, output):
    return scalar_mul(scalar_power(d_squared_error(label, output), 2), 1/2)