import matplotlib.pyplot as plt
import numpy as np
np.random.seed(3)


# Demo of Gaussian process regression on sinusoidal data, figure 6.8 in PRML.
# As Bishop's book only presents the posterior predictive distribution for a
# single data point (Eq (6.66), (6.67)), we follow the more general presentation
# in the GP book by Rasmussen and Williams in section 2.2, and compute the full
# posterior Gaussian process.

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

num_train = 7
X = np.random.rand(num_train)     # random generate training data points
sig = 0.1
eps = np.random.normal(loc=0, scale=sig, size=len(X))  # 0 mean Gaussian noise
true_t = np.sin(2 * np.pi * X)  # true targets (t for targets, following Bishop's notation)
t = true_t + eps  # noisy targets; these are what we actually observe

# for evaluation/plotting
X_eval = np.arange(0, 1, 0.01)  # evenly spaced test points
true_t_eval = np.sin(2 * np.pi * X_eval)


# kernel settings
c = 0.1
kernel = lambda x, y: gaussian_kernel(x, y, c)

X_concat = np.hstack([X, X_eval])
K = gram(X_concat, kernel)  # this is the joint Gram matrix for train and test points
# get the four blocks of K
K_train = K[:num_train, :num_train]
K_train_eval = K[:num_train, num_train:]
K_eval_train = K[num_train:, :num_train]
K_eval = K[num_train:, num_train:]

inv = np.linalg.inv(K_train + sig**2 * np.eye(num_train))
K_eval_train_mult_inv = K_eval_train @ inv
post_mean = K_eval_train_mult_inv @ t   # Eq (2.22) of R & W; Eq (6.66) of Bishop
post_cov = K_eval - K_eval_train_mult_inv @ K_train_eval  # Eq (2.23) of R & W; Eq (6.67) of Bishop
post_std = np.sqrt(np.diag(post_cov))

fig = plt.figure(figsize=(8, 6))

plt.plot(X_eval, true_t_eval, color='green', label='ground truth function')
plt.scatter(X, t, s=80, facecolors='none', edgecolors='b', label='noisy data observations')
plt.plot(X_eval, post_mean, color='r', label='posterior GP mean')
plt.fill_between(X_eval, post_mean-2*post_std, post_mean+2*post_std, color='r', alpha=0.1)

# plot a few posterior samples
samples = np.random.multivariate_normal(mean=post_mean, cov=post_cov, size=3)
for sample in samples:
    plt.plot(X_eval, sample, alpha=0.2, color='gray', label='sample from posterior GP')
plt.legend(loc='best')
plt.title('GP regression with noisy data, Figure 6.8')

plt.savefig(__file__.split('.')[0] + '.png')

