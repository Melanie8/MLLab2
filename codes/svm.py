__author__ = 'melaniesedda'
import numpy as np
import pylab
import random
from kernels import *
from cvxopt.solvers import qp
from cvxopt.base import matrix

def svm(kernel_function, args, C):
    # Generate data
    classA = [(random.normalvariate(-1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(5)] + \
             [(random.normalvariate(1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(5)]
    classB = [(random.normalvariate(0.0, 0.5), random.normalvariate(-0.5, 0.5), -1.0) for i in range(10)]
    data = classA + classB
    random.shuffle(data)
    pylab.hold(True)
    pylab.plot([p[0] for p in classA], [p[1] for p in classA], 'bo')
    pylab.plot([p[0] for p in classB], [p[1] for p in classB], 'ro')
    N = np.shape(data)[0]
    print N

    # Create matrices (rajouter des matrix partout)
    P = [[p[2]*q[2]*kernel_function(p, q, args) for p in data] for q in data]
    q = -1*np.ones(N)
    G = -1*np.identity(N)
    h = np.zeros(N)

    # Solve the optimization problem
    r = qp(matrix(P) , matrix(q) , matrix(G) , matrix(h))
    alpha = list(r['x'])

    # Plot the boundary
    xrange = np.arange(-4, 4, 0.05)
    yrange = np.arange(-4, 4, 0.05)
    grid = matrix([[indicator([x, y], alpha, data, kernel_function, args) for y in yrange] for x in xrange])
    pylab.contour(xrange, yrange, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
    pylab.show()


if __name__ == "__main__":
    svm(rbf_kernel, [2, 3], 0)
