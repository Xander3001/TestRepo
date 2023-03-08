''' Implementation of Softmax and ReLU activation functions with their respective backward functions for neural networks.

    Args:
        z (numpy array): input to the activation function.

    Returns:
        The output of the activation function and a cache for storing the input z.

    Raises:
        None
'''
import numpy as np
import matplotlib.pyplot as plt


def softmax(z):
    cache = z
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z), axis=1))
    return sm, cache


''' Implementation of ReLU activation function.

    Args:
        z (numpy array): input to the activation function.

    Returns:
        The output of the activation function and a cache for storing the input z.

    Raises:
        None
'''
def relu(z):

    s = np.maximum(0, z)
    cache = z
    return s, cache


''' Implementation of the backward function for Softmax activation.

    Args:
        dA (numpy array): derivative of the loss function with respect to output of Softmax.
        cache (tuple): a tuple containing input to Softmax.

    Returns:
        The derivative of the loss function with respect to inputs to Softmax.

    Raises:
        None
'''

def softmax_backward(dA, cache):

    z = cache
    z -= np.max(z)
    s = (np.exp(z).T / np.sum(np.exp(z), axis=1))
    dZ = dA * s * (1 - s)
    return dZ


''' Implementation of the backward function for ReLU activation.

    Args:
        dA (numpy array): derivative of the loss function with respect to output of ReLU.
        cache (tuple): a tuple containing input to ReLU.

    Returns:
        The derivative of the loss function with respect to inputs to ReLU.

    Raises:
        None
'''
def relu_backward(dA, cache):
 
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
    dZ[Z <= 0] = 0
    return dZ