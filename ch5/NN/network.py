import random
import numpy as np
from scipy.special import expit as sigmoid

# A generic feedforward neural network; backprop implemented using Bishop's equations.
# based on https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py, except using
# a proper softmax output layer (rather than sigmoid layer) for classification

class Network(object):
    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first (input) layer containing 2 neurons, the second layer 3
        neurons, and the third layer 1 neuron.  The biases and weights
        for the network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.
        We follow the convention in Bishop's book of incorporating
        bias parameters into the weights by defining an additional
        dummy input unit z_0 = 1 for each layer. """

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = [np.hstack((np.random.randn(y, 1), np.random.randn(y, x)/np.sqrt(x)))    # init bias separately
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, t)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                eval = self.evaluate(test_data)
                print("Epoch {} : {} / {}, loss = {}".format(j, eval[0], n_test, eval[1]))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, t)``, and ``eta``
        is the learning rate."""

        batch_grad = [np.zeros(w.shape) for w in self.weights]
        for x, t in mini_batch:
            grad = self.backprop(x, t)  # gradient estimate from one data point
            batch_grad = [w + dw for w, dw in zip(batch_grad, grad)]  # accumulate
        self.weights = [w - (eta / len(mini_batch)) * g
                        for w, g in zip(self.weights, batch_grad)]

    def feedforward(self, z):
        """Return the output of the network given input; "z" is a Dx1 numpy array representing outputs
        from previous layer (or simply the data input, when there's no previous layer)
        """

        for i,w in enumerate(self.weights):
            z = prepend_bias_unit(z)  # dummy input for bias
            a = np.dot(w, z)  # activation (i.e. weighted inputs), (5.2/5.8)
            if i == len(self.weights)-1: # last layer
                z = softmax(a)
            else:
                z = ReLU(a)
        return z

    def backprop(self, x, t):
        """Return a list of ndarrays the same shapes as those of self.weights
        containing the partial derivatives w.r.t weights, i.e. the gradient of
        the cost function C_x. x is a column (input) vector, t is a vector （one-hot) target"""

        # feedforward
        z = x
        zs = []  # list of layer outputs (including the "outputs" of the 0th (network input) layer
        activations = []    # list of weighted inputs (i.e. activations)
        for i,w in enumerate(self.weights):
            z = prepend_bias_unit(z)  # dummy input for bias
            zs.append(z)
            a = np.dot(w, z)  # activation (i.e. weighted inputs), (5.2/5.8)
            if i == len(self.weights) - 1: # last layer uses softmax
                z = softmax(a)
            else:
                z = ReLU(a)
            activations.append(a)

        # backwards error propagation
        y = z  # network (last layer) output
        delta = last_activation_error(z, t)  # derivatives with respect to the last (linear) activation (5.54)
        deltas = [delta]    # list of derivatives w.r.t activations
        for w, a in zip(self.weights[::-1], activations[-2::-1]):
            back_derivatives = ReLU_prime(a)  # using the previous layer's activation derivative
            back_errors = np.dot(w.T[1:], delta)  # bias weight is not involved in error inner-prod computation
            delta = back_derivatives * back_errors  # (5.56); a vector of delta_js for hidden units
            deltas.append(delta)

        # calculate all the partial derivatives w.r.t to weights going forward
        partials = []  # a list of matrices the same size as self.weights containing partial derivatives
        deltas.reverse()
        for d, z in zip(deltas, zs):  # multiply layer output error by layer input to get partial derivatives
            partials.append(np.outer(d, z))
        return partials

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result, as well as total loss. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_loss = test_num_correct = 0
        for (x, t) in test_data:
            pred = self.feedforward(x)
            test_loss += -np.log(pred[t])   # cross-entropy cost for categorical output
            test_num_correct += int(np.argmax(pred) == t)

        # test_results = [np.argmax(self.feedforward(x)) for (x, y) in test_data]
        # return sum(int(x == y) for (x, y) in test_results)
        return test_num_correct, test_loss


#### Miscellaneous functions
def sigmoid_prime(z):
    """Derivative of the logistic sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))

def prepend_bias_unit(a):
    """We fold the bias parameters into weights, so activation computed from each layer (except output layer)
    needs to be prefixed with a dummy input unit clamped at 1"""
    return np.vstack(([1], a))  # prepend bias unit as the zeroth element

def last_activation_error(y, t):
    """Compute cost derivatives with respect to the last activation dLdz; y is the vector of class scores (unnormalized
    log probabilities, y is the output vector of normalized class probabilities, t is the target 10x1 vector. """
    return y - t  # works for canonical response outputs (e.g. softmax/logistic/identity outputs with cross-entropy)

def ReLU(z):
    """Apply rectified linear unit on given vector"""
    return np.maximum(0., z)

def ReLU_prime(z):
    """Element-wise derivatives of ReLU with respect to input vector"""
    result = np.zeros_like(z)
    result[z>0] = 1
    return result

def softmax(z):
    """Compute the softmax of a vector (containing unnormalized log likelihood"""
    z-=z.max()  # to avoid numeric problems with exp
    return np.exp(z)/np.sum(np.exp(z))
