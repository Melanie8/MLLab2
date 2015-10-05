__author__ = 'melaniesedda'
import numpy as np
import pylab
from generate_data import *
from kernels import *
#from cvxopt.solvers import qp
#from cvxopt.base import matrix

def svm(kernel_function, args, C):
    # Generate data
    data = generate_random_data()
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
    xrange = numpy.arange(-4, 4, 0.05)
    yrange = numpy.arange(-4, 4, 0.05)
    grid = matrix([[indicator(x, y) for y in yrange] for x in xrange])
    pylab.contour(xrange, yrange, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))


if __name__ == "__main__":
    svm(linear_kernel, [0], 0)