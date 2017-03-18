import numpy as np
import network

def mnist():
    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 0.03, test_data=test_data)

mnist()
