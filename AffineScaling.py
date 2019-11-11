import numpy as np


class AffineScaling():

    # Consider standard form:
    # min c'x
    # s.t. Ax = b, x>=0
    def __init__(self, A, b, c, x, epsilon=1e-2, beta=0.1, trace=False):
        (m, n) = A.shape

        # Input shape check:
        if A.shape != (b.shape[0], c.shape[0]):
            raise RuntimeError("Input shape incorrect!")
        if beta <= 0 or beta >= 1:
            raise RuntimeError("beta must between (0,1)")
        if epsilon <= 0:
            raise RuntimeError("epsilon must be positive")
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
        self.X_k = np.diag(x[:, 0])
        self.p_k = None
        self.r_k = None
        self.epsilon = epsilon  # optimality threshold
        self.beta = beta
        self.trace = True
        if trace:
            self.traces = np.empty((1,n))

    # reached optimal: return true
    def __OptimalityCheck(self):
        distance = np.sum(self.X_k @ self.r_k)
        #  dual feasibliity and optimality
        return distance > 0 and distance < self.epsilon

    # unbounded: return true
    def __UnboundenessCheck(self):
        reduced_cost = - self.X_k @ self.X_k @ self.r_k
        return np.all(np.greater_equal(reduced_cost, 0))

    def __Caculate_r(self):
        self.p_k = np.linalg.inv(self.A @ self.X_k @ self.X_k @ np.transpose(
            self.A)) @ self.A @ self.X_k @ self.X_k @ self.c
        self.r_k = self.c - np.transpose(self.A) @ self.p_k

    def __Update_X(self):
        move = self.X_k @ self.X_k @ self.r_k / \
            self.__gamma(self.X_k @ self.r_k)
        move = np.diag(move[:, 0])
        self.X_k = self.X_k - self.beta * move
    
    def __gamma(self, input):
        clipped = np.clip(input, 0, None)
        return np.max(clipped)

    def Run(self):
        self.__Caculate_r()
        while not self.__OptimalityCheck():
            if self.__UnboundenessCheck():
                print("the input LP problem is unbounded")
                return None
            self.__Update_X()
            self.__Caculate_r()
            if self.trace:
                expanded_X = np.expand_dims(np.diag(self.X_k),axis=0)
                self.traces = np.concatenate((self.traces,expanded_X),axis=0)
        return np.diag(self.X_k)
    
    def GetTraces(self):
        return self.traces


def StandardFormTransformer(A_origin, b, c_origin, x_origin):
    (m, _) = A_origin.shape
    A_auxiliary = np.eye(m)
    A = np.concatenate((A_origin, A_auxiliary), axis=1)
    x_auxiliary = b - A_origin @ x_origin
    x = np.concatenate((x_origin, x_auxiliary), axis=0)
    c_auxiliary = np.zeros((m, 1))
    c = np.concatenate((c_origin, c_auxiliary), axis=0)
    return A, b, c, x


if __name__ == "__main__":

    # LP problem in standard
    epsilon = 0.1
    A_origin = np.asarray([[1, 0], [-1, 0], [epsilon, -1], [epsilon, 1]])
    b = np.asarray([[1, -epsilon, 0, 1]]).T
    c_origin = np.asarray([[0, -1]]).T
    x_origin = np.asarray([[0.5, 0.5]]).T
    (A, _, c, x) = StandardFormTransformer(A_origin, b, c_origin, x_origin)

    solver = AffineScaling(A, b, c, x, trace=True)
    result = solver.Run()
    traces = solver.GetTraces()
    print(result)
    x, y = traces[:,:2].T
    import matplotlib.pyplot as plt
    plt.scatter(x,y)
    plt.show()

