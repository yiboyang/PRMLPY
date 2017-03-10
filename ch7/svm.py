import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


# demo of soft-margin SVM based on PRML 7.1
# learn a non-linear decision boundary with Gaussian kernel
# we solve the dual problem with scipy's SLSQP implementation


def gaussian_kernel(x, y, c):
    """a.k.a. RBF, parameter c controls kernel width"""
    return np.exp(-np.sum(np.square(x - y)) / c)


def gram(X, k):
    """compute the Gram matrix, given a data matrix X and kernel k; K^2 time complexity"""
    N = len(X)
    K = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = k(X[i], X[j])

    return K


def predict(test, X, t, k, a, b):
    """Form predictions on a test set.
    :param test: matrix of test data
    :param X: matrix of training data
    :param t: vector of training labels
    :param k: kernel used
    :param a: optimal dual variables (weights)
    :param b: optimal intercept
    """
    a_times_t = a * t
    y = np.empty(len(test))  # y is the array of predictions
    for i, s in enumerate(test):  # (7.13); we skip kernel evaluation if a training data is not a support vector
        # evaluate the kernel between new data point and support vectors; 0 if not a support vector
        kernel_eval = np.array([k(s, x_m) if a_m > 0 else 0 for x_m, a_m in zip(X, a)])
        y[i] = a_times_t.dot(kernel_eval) + b
    return y


# let's learn the XOR function in R^2
N = 100
X = np.array(list(zip(np.random.normal(size=N), np.random.normal(size=N))))  # standard Gaussian samples in R^2
c0 = np.where(np.prod(X, axis=1) < 0)  # indices of points (x,y) for which XOR(x,y) is true, i.e. (x<0&y>0)|(x>0&y<0)
c1 = np.where(np.prod(X, axis=1) > 0)  # indices of data points belonging to the other class
t = np.ones(N)  # labels
t[c1] = -1  # negative class labels

# set up the convex optimization (QP) problem; the most important settings are the regularizer and the kernel; should be
# set by cross validation
C = 5  # regularizer; the greater, the more stringent we are with the amount of slack (and care more about accuracy)
kernel = lambda x, y: gaussian_kernel(x, y, 2)  # kernel width controls decision smoothness

K = gram(X, kernel)
tKt = t * K * t[:, np.newaxis]  # Hadamard product (quadratic form) between t and K; also the Hessian of (7.32)

# we'd like to minimize the negative of the dual function (7.32) subject to linear constraints (7.33) and (7.34)
# write the constraints in matrix notation: inequalities Ax <= b, equalities cx = d (`x` is our `a`); these's only one
# equality constraint, i.e. t.T.dot(a) = 0
# inequalities: A is M by N, where M=2N is the number of inequalities (N box constraints, 2 inequalities each), (7.33)
# first half for -a<=0, second half for a<=C
A = np.vstack((-np.eye(N), np.eye(N)))
b = np.concatenate((np.zeros(N), C * np.ones(N)))


def loss(a):
    """Evaluate the negative of the dual function at `a`.
    We access the optimization data (Gram matrix and target vector) from outer scope for convenience.
    :param a: dual variables
    """
    # at = a * t  # Hadamard product
    # return -(a.sum() - 0.5 * np.dot(at.T, np.dot(K, at)))  # negative of (7.32)
    # equivalent to the above; this is the canonical quadratic form involving a.T * tKt * a
    return -(a.sum() - 0.5 * np.dot(a.T, np.dot(tKt, a)))  # negative of (7.32)


def jac(a):
    """Calculate the Jacobian of the loss function (for the QP solver)"""
    return np.dot(a.T, tKt) - np.ones_like(a)


constraints = ({'type': 'ineq', 'fun': lambda x: b - np.dot(A, x), 'jac': lambda x: -A},
               {'type': 'eq', 'fun': lambda x: np.dot(x, t), 'jac': lambda x: t})
# training
a0 = np.random.rand(N)  # initial guess
print('Initial loss: ' + str(loss(a0)))
res = minimize(loss, a0, jac=jac, constraints=constraints, method='SLSQP', options={})

print('Optimized loss: ' + str(res.fun))
a = res.x  # optimal Lagrange multipliers
a[np.isclose(a, 0)] = 0  # zero out nearly zeros
a[np.isclose(a, C)] = C  # round the ones that are nearly C

# points with a==0 do not contribute to prediction
support_idx = np.where(0 < a)[0]  # index of points with a>0; i.e. support vectors
margin_idx = np.where((0 < a) & (a < C))[0]  # index of support vectors that happen to lie on the margin, with 0<a<C
print('Total %d data points, %d support vectors' % (N, len(support_idx)))

# still need to find the intercept term, b (unfortunately this name collides with the earlier Ax=b); we use (7.37)
a_times_t = a * t
support_a_times_t = (a * t)[support_idx]  # a*t for support vectors
cum_b = 0
for n in margin_idx:
    x_n = X[n]
    # evaluate the kernel between x_n and support vectors; fill the rest with zeros
    kernel_eval = np.array([kernel(x_m, X[n]) if a_m > 0 else 0 for x_m, a_m in zip(X, a)])
    b = t[n] - a_times_t.dot(kernel_eval)
    cum_b += b
b = cum_b / len(margin_idx)

# points with a==C may be misclassified (these are the only points that may be misclassified)
possible_wrong_idx = np.where(a == C)[0]
possibly_wrong_predictions = predict(X[possible_wrong_idx], X, t, kernel, a, b)
num_wrong = np.sum((possibly_wrong_predictions * t[possible_wrong_idx]) < 0)
print('Classification accuracy: ' + str(1 - num_wrong / N))

# plots
# plot training data points
for cls, clr in zip((c0, c1), ('r', 'b')):
    plt.scatter(X[cls, 0], X[cls, 1], color=clr)
# add circles around support vectors
plt.scatter(X[support_idx, 0], X[support_idx, 1], color='g', s=100, facecolors='none', edgecolors='g',
            label='support vectors')

# plot the decision boundary and margins in the input space
grid = np.arange(X.min(), X.max(), 0.05)
xx, yy = np.meshgrid(grid, grid)
zs = predict(np.array(list(zip(np.ravel(xx), np.ravel(yy)))), X, t, kernel, a, b)
zz = zs.reshape(xx.shape)
CS = plt.contour(xx, yy, zz, levels=[-1, 0, 1], )  # margin, separating hyperplane, margin
plt.clabel(CS, fmt='%2.1d', colors='k')
plt.xlabel("x_1")
plt.ylabel("x_2")
plt.legend(loc='best')
plt.title("SVM Classification of XOR Data with Gaussian Kernel")
plt.savefig(__file__.split('.')[0] + '.png')
