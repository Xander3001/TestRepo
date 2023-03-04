# This code contains four functions for implementing the softmax activation function, ReLU activation function,
# and their respective backward propagation functions.

import numpy as np
import matplotlib.pyplot as plt


def softmax(z):
    """
    Compute softmax function for input array "z".

    Arguments:
    z -- input array

    Returns:
    sm -- softmax of input array
    cache -- input array "z"
    """
    cache = z
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z), axis=1))
    return sm, cache


def relu(z):
    """
    Compute ReLU function for input array "z".

    Arguments:
    z -- input array

    Returns:
    s -- output of ReLU function for input array "z"
    cache -- input array "z"
    """
    s = np.maximum(0, z)
    cache = z
    return s, cache


def softmax_backward(dA, cache):
    """
    Compute the derivative of the softmax function for input array "dA".

    Arguments:
    dA -- input array
    cache -- input array used for softamx calculation

    Returns:
    dZ -- derivative of softmax for input array "dA"
    """
    z = cache
    z -= np.max(z)
    s = (np.exp(z).T / np.sum(np.exp(z), axis=1))
    dZ = dA * s * (1 - s)
    return dZ


def relu_backward(dA, cache):
    """
    Compute the derivative of the ReLU function for input array "dA".

    Arguments:
    dA -- input array
    cache -- input array used for ReLU calculation

    Returns:
    dZ -- derivative of ReLU for input array "dA"
    """
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
    dZ[Z <= 0] = 0
    return dZ