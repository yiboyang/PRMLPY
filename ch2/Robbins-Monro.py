import numpy as np
import matplotlib.pyplot as plt

# demo of Robbins-Monro (SGD?) algorithm for sequential estimation based on PRML 2.3.5
# in this example we estimate the mean of a Gaussian using Maximum Likelihood

N = 1000
truemu = 0
sigma = 2   # variance; known, fixed
x=np.random.normal(loc=truemu, scale=sigma, size=N)

mus = []

# initialize the estimate of the mean to some bad value
#prevmu = x[0]   # this is like the theta^(N-1) in (2.135)
prevmu = truemu+2   # intentionally bad initial guess

for n in range(2, N):
    a = 1/(n-1)   # prev step size
    z = -(x[n]-prevmu)/sigma**2 # prev z; (2.136); too bad x[1] was wasted
    mu = prevmu - a * z
    prevmu = mu
    mus.append(mu)

plt.plot(np.arange(len(mus)), mus)
