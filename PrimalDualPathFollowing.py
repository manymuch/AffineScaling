import numpy as np
from HelperFunction import LinProgBaseClass, StandardFormTransformer

class PrimalDualPathFollowing(LinProgBaseClass):

    # epsilon is optimality threshold
    def __init__(self, A, b, c, x, s, p,
                 epsilon=1e-2, rou=.6, alpha=0.6,trace=False):
        super(PrimalDualPathFollowing, self).__init__(A, b, c, x, trace=trace)

        if alpha <= 0 or alpha >= 1:
            raise RuntimeError("alpha must between (0,1)")
        if epsilon <= 0:
            raise RuntimeError("epsilon must be positive")

        self.X_k = np.diag(x[:, 0])
        self.s_k = s
        self.p_k = p
        
        self.epsilon = epsilon
        self.rou = rou
        self.alpha = alpha

    # Return True if reached optimality
    def __OptimalityCheck(self):
        column_x = np.expand_dims(np.diag(self.X_k),axis=1)
        offset = (self.s_k.T @ column_x)[0,0]
        return offset < self.epsilon
    
    def __ComputeNewtonDirections(self):
        column_x = np.expand_dims(np.diag(self.X_k),axis=1)
        offset = (self.s_k.T @ column_x)[0,0]
        miu = self.rou * offset / self.n
        Dk = np.diag(np.sqrt(column_x / self.s_k)[:, 0])
        Pk = Dk @ self.A.T @ np.linalg.inv(self.A @ Dk @ Dk @ self.A.T) @ self.A @ Dk
        vmiu = np.linalg.inv(self.X_k) @ Dk @ (miu - column_x * self.s_k)
        d_x = Dk @ (np.eye(self.n) - Pk) @ vmiu
        d_p = - np.linalg.inv(self.A @ Dk @ Dk @ self.A.T) @ self.A @ Dk @ vmiu
        d_s = np.linalg.inv(Dk) @ Pk @ vmiu

        return d_x, d_p, d_s
    
    def __Update(self, d_x, d_p, d_s):
        column_x = np.expand_dims(np.diag(self.X_k),axis=1)
        beta_p = np.min([self.alpha * np.min((- column_x / d_x)[d_x < 0]), 1])
        beta_d = np.min([self.alpha * np.min((- self.s_k / d_s)[d_s < 0]), 1])
        column_x = column_x + beta_p * d_x
        self.X_k = np.diag(column_x[:, 0])
        self.p_k = self.p_k + beta_d * d_p
        self.s_k = self.s_k + beta_d * d_s

    def Run(self):
        while not self.__OptimalityCheck():
            d_x, d_p, d_s = self.__ComputeNewtonDirections()
            self.__Update(d_x, d_p, d_s)
            if self.trace:
                expanded_X = np.expand_dims(np.diag(self.X_k),axis=0)
                self.traces = np.concatenate((self.traces,expanded_X),axis=0)
        return np.diag(self.X_k)
    
    def GetTraces(self):
        return self.traces
        




        



