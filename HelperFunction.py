import numpy as np

class LinProgBaseClass():
    # Consider standard form:
    # min c'x
    # s.t. Ax = b, x>=0
    def __init__(self, A, b, c, x, trace=False):
        (m, n) = A.shape

        # Input shape check:
        if A.shape != (b.shape[0], c.shape[0]):
            raise RuntimeError("Input shape incorrect!")
        # Feasibility check:
        if not np.allclose(A @ x, b):
            raise RuntimeError(
                "initialization not feasible, Ax = {}\nbut b = {}".format(A @ x, b))
        if not np.all(np.greater_equal(x, 0)):
            raise RuntimeError("initialization not feasible, x must >=0")
        self.m = m  # number of equality constraints
        self.n = n  # number of valuables
        self.A = A
        self.c = c  # cost fuction

        self.trace = True
        if trace:
            self.traces = np.empty((1,n))
    
def StandardFormTransformer(A_origin, b, c_origin, x_origin):
    (m, _) = A_origin.shape
    A_auxiliary = np.eye(m)
    A = np.concatenate((A_origin, A_auxiliary), axis=1)
    x_auxiliary = b - A_origin @ x_origin
    x = np.concatenate((x_origin, x_auxiliary), axis=0)
    c_auxiliary = np.zeros((m, 1))
    c = np.concatenate((c_origin, c_auxiliary), axis=0)
    return A, b, c, x