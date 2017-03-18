import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
from scipy.stats import multivariate_normal


# demo of kernel density estimator based on PRML 2.5.1

def Parzen_kernel(u):
    # Parzen window, i.e. unit hypercuber indicator (2.247)
    # input: D-dim array of data
    dshape = u.shape
    assert (len(dshape) <= 2)
    if len(dshape) == 1:
        axis = 0
    else:
        axis = 1
    return np.all(abs(u) <= 0.5, axis=axis)


def Gaussian_kernel(u):
    # Gaussian kernel, standard normal Gaussian in D-dim
    # input: D-dim array of data
    dshape = u.shape
    assert (len(dshape) <= 2)
    if len(dshape) == 1:
        D = len(u)
    else:
        D = dshape[1]
    mean = np.zeros(D)
    cov = np.eye(D)
    return multivariate_normal.pdf(u, mean=mean, cov=cov)


def kernel_density(x, X, k, h):
    # input:
    # x: single data point for which a kernel density estimate is calculated
    # X: observations (neighbors)
    # k: kernel function
    # h: bandwidth for the kernel (length of the side of hypercube)
    N, D = X.shape
    # K = 0   # "count" of observations in the hypercube centered at x
    # for x_n in X:
    #    K += k((x-x_n)/h)

    # more efficient implementation
    K = k((x - X) / h).sum()
    return K / (N * h ** D)  # (2.246)


N = 100
D = 2  # 2D data for demo
X = np.random.rand(N, D)
plt.scatter(X[:, 0], X[:, 1])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(X.min(), X.max(), 0.05)

xx, yy = np.meshgrid(x, y)
k = Gaussian_kernel
h = 0.1  # 0.2 is much smoother
zs = np.array([kernel_density([x, y], X, k, h) for x, y in zip(np.ravel(xx), np.ravel(yy))])
zz = zs.reshape(xx.shape)

ax.plot_wireframe(xx, yy, zz)

ax.set_title("Kernel Density Estimation")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("probability density")
plt.savefig(__file__.split('.')[0] + '.png')
