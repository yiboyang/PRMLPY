import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

# demo of Bayesian linear regression based on PRML 3.3.1
# reproducing parts of Figure 3.7

D = 2
noise_std = 0.2  # std of the noise in the data, assumed known; see p 154
alpha = 2  # set the variance hyperparam of prior, see eq (3.52)


# likelihood in simple linear regression as a function of w0, w1
# (x,t) is a pair of data observation
def like(w0, w1, x, t):
    return sp.stats.norm.pdf(t, loc=(w0 + w1 * x), scale=noise_std)


# set up meshgrid
grid = np.arange(-1, 1, 0.02)
xx, yy = np.meshgrid(grid, grid)
fig = plt.figure()

# plot the prior
zs = np.array([sp.stats.multivariate_normal.pdf([x, y], np.zeros(D), np.eye(D) / alpha) for x, y in
               zip(np.ravel(xx), np.ravel(yy))])
zz_p = zs.reshape(xx.shape)

ax = fig.add_subplot(222)
plt.pcolor(xx, yy, zz_p)

# likelihood function for the first observation
x1, t1 = 0.9, 0.1  # what seems to be the first observation
zs = np.array([like(x, y, x1, t1) for x, y in zip(np.ravel(xx), np.ravel(yy))])
zz_l = zs.reshape(xx.shape)

ax = fig.add_subplot(223)
ax.pcolor(xx, yy, zz_l)

# plot the (un-normalized) posterior
zz = zz_p * zz_l

ax = fig.add_subplot(224)
ax.pcolor(xx, yy, zz)
