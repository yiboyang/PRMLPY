import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit as logistic

# demo of logistic regression based on PRML 4.3.3
# the notations here are based on M. I. Jordan's Intro to PGM book chapter 7,
# and differ from the PRML book; we use y for the array of target labels,
# mu for the expected (predicted) response defined by (4.87)

# here we use the Iris data set and try to tell setosa (class 0) from
# versicolor (class 1), using the two features sepal length (cm) and sepal
# width (cm); data from sklearn.datasets.load_iris()

# we perform classification in the original data space (no basis functions) as
# this is a easy data set (linearly separable)
# there's no regularization for simplicity, as a result there's overfitting,
# especially severe with faster methods like IRLS

X = np.array([[5.1, 3.5], [4.9, 3.], [4.7, 3.2], [4.6, 3.1], [5., 3.6],
              [5.4, 3.9], [4.6, 3.4], [5., 3.4], [4.4, 2.9], [4.9, 3.1],
              [5.4, 3.7], [4.8, 3.4], [4.8, 3.], [4.3, 3.], [5.8, 4.],
              [5.7, 4.4], [5.4, 3.9], [5.1, 3.5], [5.7, 3.8], [5.1, 3.8],
              [5.4, 3.4], [5.1, 3.7], [4.6, 3.6], [5.1, 3.3], [4.8, 3.4],
              [5., 3.], [5., 3.4], [5.2, 3.5], [5.2, 3.4], [4.7, 3.2],
              [4.8, 3.1], [5.4, 3.4], [5.2, 4.1], [5.5, 4.2], [4.9, 3.1],
              [5., 3.2], [5.5, 3.5], [4.9, 3.1], [4.4, 3.], [5.1, 3.4],
              [5., 3.5], [4.5, 2.3], [4.4, 3.2], [5., 3.5], [5.1, 3.8],
              [4.8, 3.], [5.1, 3.8], [4.6, 3.2], [5.3, 3.7], [5., 3.3],
              [7., 3.2], [6.4, 3.2], [6.9, 3.1], [5.5, 2.3], [6.5, 2.8],
              [5.7, 2.8], [6.3, 3.3], [4.9, 2.4], [6.6, 2.9], [5.2, 2.7],
              [5., 2.], [5.9, 3.], [6., 2.2], [6.1, 2.9], [5.6, 2.9],
              [6.7, 3.1], [5.6, 3.], [5.8, 2.7], [6.2, 2.2], [5.6, 2.5],
              [5.9, 3.2], [6.1, 2.8], [6.3, 2.5], [6.1, 2.8], [6.4, 2.9],
              [6.6, 3.], [6.8, 2.8], [6.7, 3.], [6., 2.9], [5.7, 2.6],
              [5.5, 2.4], [5.5, 2.4], [5.8, 2.7], [6., 2.7], [5.4, 3.],
              [6., 3.4], [6.7, 3.1], [6.3, 2.3], [5.6, 3.], [5.5, 2.5],
              [5.5, 2.6], [6.1, 3.], [5.8, 2.6], [5., 2.3], [5.6, 2.7],
              [5.7, 3.], [5.7, 2.9], [6.2, 2.9], [5.1, 2.5], [5.7, 2.8]])
N = len(X)
X = np.hstack((np.ones(N)[:, np.newaxis], X))  # add the bias feature
y = np.array([0] * 50 + [1] * 50)  # labels

M = 3  # dimension of theta (weights)


def expected_response(theta, X):
    # compute the conditional expectation of the response as defined by logistic
    # regression, i.e. E[y_n|x_n, theta] = sigmoid(theta dot x_n), which is
    # also equal to the posterior class probability P(y_n=1|x_n, theta)
    return logistic(X.dot(theta))


def error(theta, X, y):
    # compute the error (cost) associated with theta; this is the negative log
    # likelihood p(y|x, theta), and is the objective we wish to minimize
    # we compute it directly from the inner product of X and theta (a.k.a. the
    # logit) (rather than using mu=expected_response(..), then compute
    # -(y*np.log(mus) + (1-y)*np.log(1-mus))) to avoid taking logs of possibly
    # saturated sigmoid functions

    # let eta be the logit, then p(y|x) = exp(eta*y)/(1+exp(eta))
    eta = X.dot(theta)
    return np.sum(-eta * y + np.log(1 + np.exp(eta)))


def plot_results(Es, thetas, X, y):
    """Plot the cost function, parameter size across training iteration and 2D decision boundary"""
    fig = plt.figure()

    # plot the error function history
    fig.add_subplot(221)
    plt.plot(Es)
    plt.xlabel("iter")
    plt.ylabel("cost (neg log likelihood)")
    plt.title("Cost Function Across Iterations")

    # plot the size of weights
    fig.add_subplot(222)
    plt.plot(np.sum(np.square(thetas), axis=1))
    plt.xlabel("iter")
    plt.ylabel("weights size (L2 norm)")
    plt.title("Parameter Size Across Iterations")

    # plot final decision region in input space
    fig.add_subplot(223)
    # grid = np.arange(X.min(), X.max(), 0.2)
    xx, yy = np.meshgrid(np.arange(X[:,1].min(), X[:,1].max(), 0.2), np.arange(X[:,2].min(), X[:,2].max(), 0.2))
    theta = thetas[-1]  # use the parameters from last iteration
    zs = np.array([theta[0] + theta[1] * x + theta[2] * y for x, y in zip(np.ravel(xx), np.ravel(yy))])
    zz = zs.reshape(xx.shape)
    plt.contour(xx, yy, zz, [0], label="final decision boundary")  # only plot the level set z(x)=0
    plt.scatter(X[y == 0][:, 1], X[y == 0][:, 2], color='r')
    plt.scatter(X[y == 1][:, 1], X[y == 1][:, 2], color='g')
    plt.title("Logistic Regression on Iris Data")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend(loc='best')

    fig.add_subplot(224)    # empty plot for overall spacing
    fig.tight_layout()
    plt.savefig(__file__.split('.')[0] + '.png')


def bt_line_search(f, x, g, d):
    """Use backtracking line search to determine a step size for gradient descent update
    :param f: objective function to be minimized
    :param x: a point in the domain of f
    :param g: the gradient of f at x
    :param d: the proposed descent direction of f at the point x
    :return t: step size determined by backtracking line search
    More details see Boyd's Convex Optimization, p. 464
    """
    alpha = 0.1  # should be in (0, 0.5)
    beta = 0.5  # should be in (0, 1)
    t = 1
    f_x = f(x)
    g_dot_d = g.dot(d)
    while f(x + t * d) > f_x + alpha * t * g_dot_d:  # Armijo-Goldstein condition
        t *= beta
    return t


def sgd(X, y, num_epochs, rho, plot=True):
    """Stochastic gradient descent training
    :param rho: learning rate (the fixed descent step size)
    """
    # parameters to be learned
    theta = np.random.rand(M) - 0.5  # random in [-1,1)
    # data indices
    idx = np.arange(N)

    Es = []  # history of errors
    thetas = []  # history of thetas

    for ep in range(num_epochs):
        np.random.shuffle(idx)
        for n in idx:  # iterate through data
            x_n, y_n = X[n], y[n]
            mu_n = expected_response(theta, x_n)
            direction = (y_n - mu_n) * x_n  # stochastic estimate negative gradient
            theta += rho * direction

            thetas.append(theta.copy())
            Es.append(error(theta, X, y))

    if plot:
        plot_results(Es, thetas, X, y)


def batch_gd(X, y, num_epochs, rho=None, plot=True):
    """Batch gradient descent training
    :param rho: step-size of gradient update; if not specified, will use line search to find out
    """

    line_search = False
    if rho is None:  # makes line-search more convenient
        line_search = True
        fun = lambda theta: error(theta, X, y)  # the objective function solely in theta

    Es = []  # history of errors
    thetas = []  # history of thetas
    theta = np.random.rand(M) - 0.5  # random in [-1,1)

    for ep in range(num_epochs):
        # calculate gradient by going through all data
        mus = expected_response(theta, X)

        # choose the descent direction; it happens that we (somewhat naively) choose
        # it to be the negative gradient, i.e. steepest descent direction in L2 norm;
        # we could have chosen it by using the quadratic norm defined
        # by the Hessian at the current point (which corresponds to Newton's method
        direction = (y - mus).dot(X)  # negative gradient of the error function

        if line_search:
            rho = bt_line_search(fun, theta, -direction, direction)

        theta += rho * direction

        thetas.append(theta.copy())
        Es.append(error(theta, X, y))

    if plot:
        plot_results(Es, thetas, X, y)


def irls(X, y, num_epochs, plot=True):
    """Training by iterative reweighted least squares"""
    Es = []  # history of errors
    thetas = []  # history of thetas
    theta = np.random.rand(M) - 0.5  # random in [-1,1)

    for ep in range(num_epochs):
        # calculate the weight matrix; use an array of diagonal elements for
        # efficiency
        mus = expected_response(theta, X)
        W = mus * (1 - mus)  # weights

        XtWX = (X.T * W).dot(X)  # inverse of the Hessian
        rhs = XtWX.dot(theta) + X.T.dot(y - mus)

        theta = np.linalg.solve(XtWX, rhs)  # update

        thetas.append(theta.copy())
        Es.append(error(theta, X, y))

    if plot:
        plot_results(Es, thetas, X, y)


# first, let's try SGD
# learning rate 1 seems too big (overshooting)

# sgd(X, y, 10, .5, True)


# let's try batch gradient descent with fixed step size
# rho (step size/ learning rate) took many trial and errors to get "right"
# anything > 0.1 is way too big; error function staying at inf, thetas are huge
# and not getting any better; 0.01 makes the error curve go zig-zagging (over-
# shooting?); 0.001 finally gives a smoothly decreasing error curve and
# reasonable decision boundaries, but is a bit too small (takes over 200 iters
# to converge); 0.005 results in some zig-zag error curve initially, but it
# eventually goes down, and converges in about 50 iterations

# batch_gd(X, y, 50, 0.005, True)


# let's try batch gradient descent, but using line-search to decide step size;
# this is "proper" gradient descent as the objective is guaranteed to decrease
# during each iteration

batch_gd(X, y, 50, None, True)

# IRLS is lightning fast compared to the previous algorithms;
# reaches the optimal solution in < 5 iterations; overfitting starts afterwards
# (b/c this data set is linearly separable), i.e. thetas start growing without
# bound and mus get closer and closer to 0s and 1s (more and more confident)

# irls(X, y, 8, True)
