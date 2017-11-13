def squared_error(label, output):
    return [[label[i] - output[0][i] for i in range(len(output[0]))]]
