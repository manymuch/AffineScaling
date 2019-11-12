import numpy as np
from HelperFunction import LinProgBaseClass, StandardFormTransformer

class PrimalDualPathFollowing():

    # epsilon is optimality threshold
    # beta is the stepsize that controling the elipsoid size
    def __init__(self, A, b, c, x, s,
                 epsilon=1e-2, beta=0.1, rou=0.1, trace=False):
        super(PrimalDualPathFollowing, self).__init__(A, b, c, x, trace=trace)

        if beta <= 0 or beta >= 1:
            raise RuntimeError("beta must between (0,1)")
        if epsilon <= 0:
            raise RuntimeError("epsilon must be positive")
        if not np.all(np.greater_equal(s, 0)):
            raise RuntimeError("initialization not feasible, s must >=0")

        self.X_k = np.diag(x[:, 0])
        self.p_k = None
        self.s_k = None
        self.epsilon = epsilon
        self.beta = beta
        self.rou = rou

    def __OptimalityCheck(self):
        