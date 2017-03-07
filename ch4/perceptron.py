import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# demo of the perceptron learning algorithm based on PRML 4.1.7
# 2D data; using all polynomials of degree 2 or less as our bases,
# i.e. phi_{0,...,5} = (1, x_1, x_2, x_1*x_2, x1_^2, x2_^2)

def error(w, Phi, t):
    # (4.54); the objective to be minimized
    results = Phi.dot(w)*t
    # all entries greater than 0 are correctly classified and don't contribute
    results[np.where(results>0)] = 0
    E = -results.sum()
    return E

def z(w, x):
    # weighted features of an input, i.e. w.T * phi(x)
    x_1, x_2 = x
    phi = [1, x_1, x_2, x_1*x_2, x_1**2, x_2**2]
    return w.dot(phi)


X = np.array([
    # points inside the unit circle
    [0, 0],
    [0, 0.5],
    [-0.5, -0.5],
    [-0.6, 0.7],
    [0.5, -0.6],
    # points outside the unit circle
    [2, 0],
    [0.5, 1],
    [-2, -1],
    [-1, 1],
    [0, -1.5],
    ])

# distances from the origin
d = np.square(X).sum(axis=1)
# class labels; 1 if within unit circle, -1 otherwise
t = np.empty_like(d)
t[np.where(d<1)] = 1
t[np.where(d>=1)] = -1

# here we use all polynomials in x_1, x_2 of degree 2 or less as our bases,
# i.e. phi_{0,...,5} = (1, x_1, x_2, x_1*x_2, x1_^2, x2_^2)
# the basis functions is flexible enough to describe the true decision
# boundary in input space, e.g. x_1^2 + x_2^2 = 1 (which corresponds to the
# linear decision boundary phi_4 + phi_5 = 1)

# calculate the N by M matrix of features
N = len(X)
M = 6
Phi = np.empty((N,M))
# phi_0 = 1
Phi[:,0] = 1
# phi_1 = x_1
Phi[:,1] = X[:,0]
# phi_2 = x_2
Phi[:,2] = X[:,1]
# phi_3 = x_1*x_2
Phi[:,3] = X[:,0]*X[:,1]
# phi_4 = x_1^2
Phi[:,4] = np.square(X[:,0])
# phi_5 = x_2^2
Phi[:,5] = np.square(X[:,1])


# weight vector to be learned
w = np.random.rand(M)
# learning rate; set to 1 without loss of generality (p. 194)
eta = 1
# data indices
idx = np.arange(N)
num_epochs = 10

Es = []     # history of errors
ws = []    # history of weights


for ep in range(num_epochs):
    np.random.shuffle(idx)
    for n in idx:   # iterate through data
        p_n, t_n = Phi[n], t[n]
        y = 1 if w.dot(p_n) >= 0 else -1     # (4.52), (4.53); prediction
        if y == t_n:    # no change
            pass
        else:   # misclassification
            w += eta*p_n*t_n

        ws.append(w.copy())
        Es.append(error(w,Phi,t))


grid=np.arange(-2,2,0.05)
xx, yy = np.meshgrid(grid,grid)
fig = plt.figure()

# plot errors
ax = fig.add_subplot(121)
plt.plot(Es)

# plot final decision region in input space (an ellipse)
ax = fig.add_subplot(122)
plt.scatter(X[:5,0],X[:5,1], color='r')
plt.scatter(X[5:,0],X[5:,1], color='b')
zs = np.array([z(w,[x,y]) for x,y in zip(np.ravel(xx), np.ravel(yy))])
zz = zs.reshape(xx.shape)
plt.contour(xx,yy,zz,[0])   # only plot the level set z(x)=0

