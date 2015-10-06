__author__ = 'melaniesedda'
import numpy as np
import pylab
import random
from kernels import *
from cvxopt.solvers import qp
from cvxopt.base import matrix

def svm(classA, classB, kernel_function, args, C):
    # Shuffle data
    data = classA + classB
    random.shuffle(data)
    pylab.hold(True)
    pylab.plot([p[0] for p in classA], [p[1] for p in classA], 'bo')
    pylab.plot([p[0] for p in classB], [p[1] for p in classB], 'ro')
    N = np.shape(data)[0]

    # Create matrices (rajouter des matrix partout)
    P = [[p[2]*q[2]*kernel_function(p[0:2], q[0:2], args) for p in data] \
         for q in data]
    print P
    q = -1*np.ones(N)
    G = -1*np.identity(N)
    h = np.zeros(N)

    # Solve the optimization problem
    r = qp(matrix(P) , matrix(q) , matrix(G) , matrix(h))
    alpha = list(r['x']) 
    for i in range(len(alpha)):
        if alpha[i] <= math.pow(10, -5):
	    alpha[i] = 0
    print [data[i] for i in range(len(data)) if alpha[i] != 0]

    # Plot the boundary
    xrange = np.arange(-4, 4, 0.05)
    yrange = np.arange(-4, 4, 0.05)
    grid = matrix([[indicator([x, y], alpha, data, kernel_function, args) \
                    for y in yrange] for x in xrange])
    pylab.contour(xrange, yrange, grid, (-1.0, 0.0, 1.0), \
                  colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
    pylab.show()


if __name__ == "__main__":
    test = 7

    # Linear
    if test == 1:
        classA = [(-1.0, 1.0, 1.0), (-1.0, 2.0, 1.0)]
        classB = [(1.0, -1.0, -1.0), (2.0, 0.0, -1.0)]
        svm(classA, classB, linear_kernel, [0], 0)

    if test == 2:
        classA = [(random.normalvariate(-1.5, 1.0), \
                   random.normalvariate(0.5, 1.0), 1.0) for i in range(5)]
        classB = [(random.normalvariate(0.0, 0.5), \
                   random.normalvariate(-0.5, 0.5), -1.0) for i in range(10)]
        svm(classA, classB, linear_kernel, [0], 0)

    # Polynomial
    if test == 3:
        classA = [(random.normalvariate(2.0, 0.5), \
                   random.normalvariate(2.0, 0.5), 1.0) for i in range(5)] + \
                 [(random.normalvariate(0.0, 0.5), \
                   random.normalvariate(2.5, 0.5), 1.0) for i in range(5)]
        classB = [(random.normalvariate(0.0, 0.5), \
                   random.normalvariate(0.0, 0.5), -1.0) for i in range(10)]
        svm(classA, classB, polynomial_kernel, [2], 0)
	svm(classA, classB, polynomial_kernel, [3], 0)
	svm(classA, classB, polynomial_kernel, [4], 0)

    # RBF
    if test == 4:
        classA = [(random.normalvariate(2.0, 0.5), \
                   random.normalvariate(2.0, 0.5), 1.0) for i in range(5)] + \
                 [(random.normalvariate(0.0, 0.5), \
                   random.normalvariate(2.5, 0.5), 1.0) for i in range(5)]
        classB = [(random.normalvariate(0.0, 0.5), \
                   random.normalvariate(0.0, 0.5), -1.0) for i in range(10)]
        svm(classA, classB, rbf_kernel, [1], 0)
	svm(classA, classB, rbf_kernel, [2], 0)
	svm(classA, classB, rbf_kernel, [3], 0)

    if test == 5:
        classA = [(random.normalvariate(-1.5, 0.5), \
                   random.normalvariate(1.5, 0.5), 1.0) for i in range(5)] + \
                 [(random.normalvariate(1.5, 0.5), \
                   random.normalvariate(-1.5, 0.5), 1.0) for i in range(5)] + \
                 [(random.normalvariate(-1.5, 0.5), \
                   random.normalvariate(-1.5, 0.5), 1.0) for i in range(5)]
        classB = [(random.normalvariate(1.5, 0.5), \
                   random.normalvariate(1.5, 0.5), -1.0) for i in range(5)] + \
                 [(-1.0, -1.0, -1.0)]
        svm(classA, classB, rbf_kernel, [1], 0)
	svm(classA, classB, rbf_kernel, [2], 0)
	svm(classA, classB, rbf_kernel, [3], 0)

    if test == 6:
        classA = [(random.normalvariate(-1.5, 0.5), \
                   random.normalvariate(1.5, 0.5), 1.0) for i in range(5)] + \
                 [(random.normalvariate(1.5, 0.5), \
                   random.normalvariate(-1.5, 0.5), 1.0) for i in range(5)] + \
                 [(random.normalvariate(-1.5, 0.5), \
                   random.normalvariate(-1.5, 0.5), 1.0) for i in range(5)] + \
                 [(0.0, 0.0, 1.0)]
        classB = [(random.normalvariate(1.5, 0.5), \
                   random.normalvariate(1.5, 0.5), -1.0) for i in range(5)] + \
  		 [(-1.0, -1.0, -1.0)]
        svm(classA, classB, rbf_kernel, [1], 0)
	svm(classA, classB, rbf_kernel, [2], 0)
	svm(classA, classB, rbf_kernel, [3], 0)

    # Sigmoid
    if test == 7:
        classA = [(random.normalvariate(-1.5, 0.5), \
                   random.normalvariate(1.5, 0.5), 1.0) for i in range(5)] + \
                 [(random.normalvariate(1.5, 0.5), \
                   random.normalvariate(-1.5, 0.5), 1.0) for i in range(5)] + \
                 [(random.normalvariate(-1.5, 0.5), \
                   random.normalvariate(-1.5, 0.5), 1.0) for i in range(5)]
        classB = [(random.normalvariate(1.5, 0.5), \
                   random.normalvariate(1.5, 0.5), -1.0) for i in range(5)] + \
                 [(-0.5, -0.5, -1.0)]
        svm(classA, classB, sigmoid_kernel, [0.15, -0.5], 0)
	"""svm(classA, classB, sigmoid_kernel, [0.1, -0.5], 0)
	svm(classA, classB, sigmoid_kernel, [0.1, 0.0], 0)
	svm(classA, classB, sigmoid_kernel, [0.2, -1], 0)
	svm(classA, classB, sigmoid_kernel, [0.3, -1], 0)"""
