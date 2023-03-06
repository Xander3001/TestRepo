import numpy as np
import matplotlib.pyplot as plt


def softmax(z):
    """
    Computes the softmax activation function.

    Arguments:
    z -- numpy array of any shape

    Returns:
    sm -- output of softmax(z), same shape as z
    cache -- input to the softmax function (useful during backpropagation)
    """

    cache = z
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z), axis=1))
    return sm, cache


def relu(z):
    """
    Computes the ReLU activation function.

    Arguments:
    z -- numpy array of any shape

    Returns:
    s -- output of ReLU(z), same shape as z
    cache -- input to the ReLU function (useful during backpropagation)
    """

    s = np.maximum(0, z)
    cache = z
    return s, cache


def softmax_backward(dA, cache):
    """
    Computes the derivative of the softmax function.

    Arguments:
    dA -- post-activation gradient, same shape as activation (sm)
    cache -- input of the softmax function (same as output of forward propagation)

    Returns:
    dZ -- Gradient of the cost with respect to z
    """

    z = cache
    z -= np.max(z)
    s = (np.exp(z).T / np.sum(np.exp(z), axis=1))
    dZ = dA * s * (1 - s)
    return dZ


def relu_backward(dA, cache):
    """
    Computes the derivative of the ReLU function.

    Arguments:
    dA -- post-activation gradient, same shape as activation (s)
    cache -- input of the ReLU function (same as output of forward propagation)

    Returns:
    dZ -- Gradient of the cost with respect to z
    """

    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
    dZ[Z <= 0] = 0
    return dZ