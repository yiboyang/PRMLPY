import matplotlib.pyplot as plt
import numpy as np


# demo of kernel ridge regression based on PRML 6.1
# exactly what it sounds like--ridge regression with kernels; does not yield sparse solutions

def polynomial_kernel(x, y, d):
    """d-degree polynomial kernel; linear regression uses d=1"""
    return (np.dot(x, y) + 1) ** d


def gaussian_kernel(x, y, c):
    """a.k.a. RBF, parameter c controls kernel width"""
    return np.exp(-np.sum(np.square(x - y)) / c)


def gram(X, k):
    """compute the gram matrix, given a data matrix X and kernel k; K^2 time complexity"""
    N = len(X)
    K = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = k(X[i], X[j])

    return K


def predict(X, test, a, kernel):
    """Form predictions on a test set.
    :param X: matrix of training data
    :param test: matrix of test data
    :param a: optimal dual variables (weights)
    :param kernel: kernel used"""
    y = np.empty(len(test))  # y is the array of predictions
    for i, s in enumerate(test):  # eq (6.9)
        k_s = np.array([kernel(x, s) for x in X])  # the new data point's kernel evaluation with all training data
        y[i] = a.dot(k_s)
    return y


# we use the sinusoidal data set from chapter 1
X = np.arange(0, 1, 0.01)
N = len(X)
sigma = np.random.normal(loc=0, scale=0.1, size=N)  # 0 mean Gaussian noise
true_t = np.sin(2 * np.pi * X)  # true targets (t for targets, following Bishop's notation)
t = true_t + sigma  # noisy targets; these are what we actually observe

# settings
lamb = 0.001  # regularizer; smoothness penalty; needs cross-validation to get it right
kernel = lambda x, y: polynomial_kernel(x, y, 3)  # let's use 3rd degree polynomial kernel

# "training" (solution has closed form)
K = gram(X, kernel)
a = np.linalg.solve((K + lamb * np.eye(N)), t)  # eq (6.8), the optimal dual variables

# prediction
y = predict(X, X, a, kernel)  # for simplicity, let's predict the training data, just to see how well we did

# plots
plt.plot(X, true_t, color='r', label='data generating function')  # the true data generating function
plt.scatter(X, t, color='b', label='training observations')  # the noisy training data points
plt.scatter(X, y, color='c', label='predictions')  # our predictions on training data

plt.title("Kernel Ridge Regression")
plt.xlabel("input")
plt.ylabel("output")
plt.legend(loc='best')
plt.savefig(__file__.split('.')[0] + '.png')
