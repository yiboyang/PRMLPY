import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# attempt to reproduce Figure 3.10, p. 159

# N data points uniformly spaced over the interval (-1,1)
N = 200
X = np.linspace(-1,1,N)

M = 10  # number of basis functions; first one is the bias, the rest are Gaussians evenly spaced over (-1,1)
k_centers = np.linspace(-1,1,M-1)
k_bandwidth = 0.2

# compute the design matrix
Phi = np.empty((N, M))
for m in range(M):
    # compute a column at a time
    # k_vals is the m_th basis evaluated on all data
    if m == 0:
        # the bias basis
        k_vals = np.ones(N)
    else:
        k_center = k_centers[m-1]
        k_vals = np.exp(-np.square(X-k_center)/(2*k_bandwidth**2))# (3.4) p. 139
    Phi[:,m] = k_vals

alpha = 0.1
beta = 0.1  # arbitrarily set
S_N = np.linalg.inv(alpha*np.eye(M) + beta*np.dot(Phi.T, Phi))  # (3.54)


# set up meshgrid
grid=np.arange(-1,1,0.02)
gridl = len(grid)
xx, yy = np.meshgrid(grid,grid)

# evaluate the posterior mean kernel on grid points (3.62)
vals = np.empty((gridl,M))
# first evaluate the plain Gaussian kernel as before
for m in range(M):
    if m == 0:
        # the bias basis
        k_vals = np.ones(gridl)
    else:
        k_center = k_centers[m-1]
        k_vals = np.exp(-np.square(grid-k_center)/(2*k_bandwidth**2))   # (3.4), p 139.
    vals[:,m] = k_vals
# then multiply by beta and S_N
zz = beta * np.dot(np.dot(vals, S_N), vals.T)   # (3.62) gridl x gridl

fig = plt.figure()
plt.pcolor(xx,yy,zz)
