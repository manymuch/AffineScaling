import numpy as np

class AffineScaling():

# Consider standard form:
# min c'x
# s.t. Ax = b, x>=0
    def __init__(self, A,b,c,x):
        (m, n) = A.shape
        
        # Feasibility check:
        if not np.allclose(A @ x, b):
            raise RuntimeError("initialization not feasible")


def StandardFormTransformer(A_origin, b, x_origin):
    (m, _) = A_origin.shape
    A_auxiliary = np.eye(m)
    A = np.concatenate((A_origin, A_auxiliary), axis=1)
    x_auxiliary = b - A_origin @ x_origin
    x = np.concatenate((x_origin, x_auxiliary), axis=0)
    return A, b, x

if __name__ == "__main__":
    
    # LP problem in standard
    epsilon = 0.1
    A_origin = np.asarray([[1,0],[-1,0],[-epsilon, 1],[-epsilon, -1]])
    b = np.asarray([[epsilon, -1, 0, -1]]).transpose()
    c = np.asarray([[0,-1,0,0,0,0]]).transpose()
    x_origin = np.asarray([[1/2, 1/2]]).transpose()
    (A, _, x) = StandardFormTransformer(A_origin, b, x_origin)
    
    solver = AffineScaling(A,b,c,x)