import numpy as np
import network


def small():
    nn = network.Network([4,3,2])
    x = np.array([-2,-1,0,1])[:,None]
    target = np.array([0,0])[:,None]

    # initial prediction (random)
    y = nn.feedforward(x)
    print("Init cost: ", network.single_cost(y, target))

    # run one iter of SGD
    grad = nn.backprop(x, target)
    for w,g in zip(nn.weights, grad):
        w -= 0.5 * g
    y = nn.feedforward(x)
    print("Cost after training on one data point: ", network.single_cost(y, target))

def mnist():
    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

small()
