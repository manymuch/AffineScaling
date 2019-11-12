import numpy as np
from AffineScaling import AffineScaling
from HelperFunction import StandardFormTransformer
import matplotlib.pyplot as plt


eps = 0.1
offset = 0.05

# n = 3
A_origin = np.asarray([[  1,  0, 0],
                       [ -1,  0, 0],
                       [eps, -1, 0],
                       [eps,  1, 0],
                       [  0,eps,-1],
                       [  0,eps, 1]])
b = np.asarray([[1, -eps, 0, 1, 0, 1]]).T
c_origin = np.asarray([[0, 0, -1]]).T



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_zlim([0,1])


# start from point 1/2 e
x_origin = np.asarray([[0.5, 0.5, 0.5]]).T
(A, _, c, x) = StandardFormTransformer(A_origin, b, c_origin, x_origin)
solver = AffineScaling(A, b, c, x, epsilon=1e-5, trace=True)
result = solver.Run()
traces = solver.GetTraces()
x, y, z = traces[:,:3].T
ax.scatter(x,y,z, c='r', marker='.')

# start from point close to (eps, eps^2, eps^3)
x_origin = offset + np.asarray([[eps, eps**2, eps**3]]).T
(A, _, c, x) = StandardFormTransformer(A_origin, b, c_origin, x_origin)
solver = AffineScaling(A, b, c, x, epsilon=1e-5, trace=True)
result = solver.Run()
traces = solver.GetTraces()
x, y, z = traces[:,:3].T
ax.scatter(x,y,z, c='g', marker='.')

# start from point close to (eps, 1-eps^2, eps(1-eps^2))
x1 = eps+offset
x2 = 1-eps*x1-offset
x3 = 1-eps*x2-offset
x_origin = np.asarray([[x1, x2, x3]]).T
(A, _, c, x) = StandardFormTransformer(A_origin, b, c_origin, x_origin)
solver = AffineScaling(A, b, c, x, epsilon=1e-5, trace=True)
result = solver.Run()
# print(result)
traces = solver.GetTraces()
x, y, z = traces[:,:3].T
ax.scatter(x,y,z, c='b', marker='.')

# start from point close to (1,1,0) farest from goal
x_origin = np.asarray([[0.8, 0.8, 0.1]]).T
(A, _, c, x) = StandardFormTransformer(A_origin, b, c_origin, x_origin)
solver = AffineScaling(A, b, c, x, epsilon=1e-5, trace=True)
result = solver.Run()
traces = solver.GetTraces()
x, y, z = traces[:,:3].T
ax.scatter(x,y,z, c='y', marker='.')

plt.show()
