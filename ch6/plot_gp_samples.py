import matplotlib.pyplot as plt
import numpy as np


# Demo of Gaussian process samples in Figure 6.4 in PRML

def gaussian_kernel(x, y, c):
    """a.k.a. RBF, parameter c controls kernel width"""
    return np.exp(-np.sum(np.square(x - y)) / c)

def exponential_kernel(x, y, t):
    """ Eq. (6.56) """
    return np.exp(-t * np.linalg.norm(x - y))

def gram(X, k):
    """compute the Gram matrix, given a data matrix X and kernel k; K^2 time complexity"""
    N = len(X)
    K = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = k(X[i], X[j])

    return K



X = np.arange(0, 1, 0.01)
N = len(X)


# kernel settings
c = 0.1
t = 1
gp_mean = np.zeros(N)

kernels = [lambda x, y: gaussian_kernel(x, y, c),
        lambda x, y: exponential_kernel(x, y, t)]
fig = plt.figure(figsize=(8, 4))
for i in range(2):
    kernel = kernels[i]
    plt.subplot(1, 2, i + 1)
    K = gram(X, kernel) # this provides the covariance matrix of f(X)
    f_X_samples = np.random.multivariate_normal(mean=gp_mean, cov=K, size=5)
    for sample in f_X_samples:
        plt.plot(X, sample)

fig.suptitle("Samples from two Gaussian processes, Fig 6.4")
plt.savefig(__file__.split('.')[0] + '.png')

