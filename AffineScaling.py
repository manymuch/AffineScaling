import numpy as np

class AffineScaling():

# Consider standard form:
# min c'x
# s.t. Ax = b, x>=0
    def __init__(self, A,b,c,x):
        m = A.shape()
        print(m)




if __name__ == "__main__":
    
    # LP problem in standard
    epsilon = 0.1
    A = np.asarray([[1,0,1,0,0,0],[-1,0,0,1,0,0],[-epsilon,1,0,0,1,0],[-epsilon,-1,0,0,0,1]])