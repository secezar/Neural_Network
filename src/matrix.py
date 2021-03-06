from operator import mul, add, sub


def dot(a, b):
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise ValueError("Dimension mismatch: {}x{} is not {}x{}".format(len(a), len(a[0]), len(b), len(b[0])))
    return [list(map(mul, a[i], b[i])) for i in range(len(a))]


def multiply(a, b):
    zip_b = list(zip(*b))
    return [[sum(elem_a * elem_b for elem_a, elem_b in zip(row_a, col_b))
             for col_b in zip_b] for row_a in a]


def transpose(a):
    return [list(x) for x in zip(*a)]


def row_sum(a):
    return [sum(row) for row in a]


def matrix_plus(a, b):
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise ValueError("Dimension mismatch: {}x{} is not {}x{}".format(len(a), len(a[0]), len(b), len(b[0])))
    return [list(map(add, a[i], b[i])) for i in range(len(a))]


def matrix_sub(a, b):
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise ValueError("Dimension mismatch: {}x{} is not {}x{}".format(len(a), len(a[0]), len(b), len(b[0])))
    return [list(map(sub, a[i], b[i])) for i in range(len(a))]


def matrix_square(a, b):
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise ValueError("Dimension mismatch: {}x{} is not {}x{}".format(len(a), len(a[0]), len(b), len(b[0])))
    return [list(map(pow, a[i], b[i])) for i in range(len(a))]


def scalar_plus(a, val):
    added = [[col + val for col in row] for row in a]
    return added


def scalar_minus(a, val):
    subtracted = [[col - val for col in row] for row in a]
    return subtracted


def scalar_mul(a, val):
    multiplied = [[col * val for col in row] for row in a]
    return multiplied


def scalar_power(a, val):
    powered = [[col ** val for col in row] for row in a]
    return powered

def mean(vector):
    return sum(vector)/len(vector)
