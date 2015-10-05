__author__ = 'melaniesedda'
import math
import numpy as np


def linear_kernel(x, y, args):
    return np.dot(x, y) + 1


def polynomial_kernel(x, y, args):
    return math.pow(np.dot(x, y), args[0]) + 1


def rbf_kernel(x, y, args):
    return math.exp(-1/(2*math.pow(args[0], 2)*math.pow(np.linalg.norm(x-y), 2)))


def sigmoid_kernel(x, y, args):
    return math.tanh(args[0]*np.dot(x, y) - args[1])


if __name__ == "__main__":
    x = np.array([1, 2])
    y = np.array([3, 4])
    print linear_kernel(x, y)
    print polynomial_kernel(x, y, 2)
    print rbf_kernel(x, y, 2)
    print sigmoid_kernel(x, y, 2, 5)