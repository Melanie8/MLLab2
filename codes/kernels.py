__author__ = 'melaniesedda'
import math
import numpy as np


def linear_kernel(x, y, args):
    return np.dot(x, y) + 1


def polynomial_kernel(x, y, args):
    return math.pow(np.dot(x, y), args[0]) + 1


def rbf_kernel(x, y, args):
    return math.exp(-1/(2*math.pow(args[0], 2))*math.pow(np.linalg.norm([x[i]-y[i] for i in range(len(x))]), 2))


def sigmoid_kernel(x, y, args):
    return math.tanh(args[0]*np.dot(x, y) - args[1])

def indicator(x, alpha, data, kernel, args):
    return sum([alpha[i]*data[i][2]*kernel(x, data[i][0:2], args) for i in range(len(alpha))])

if __name__ == "__main__":
    x = np.array([1, 2])
    y = np.array([3, 4])
    print linear_kernel(x, y)
    print polynomial_kernel(x, y, 2)
    print rbf_kernel(x, y, 2)
    print sigmoid_kernel(x, y, 2, 5)
