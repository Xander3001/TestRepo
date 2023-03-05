import numpy as np
import matplotlib.pyplot as plt


def softmax(z):
    """
    Compute the softmax activation function for a given input vector.

    Args:
        z (numpy.ndarray): Input vector.

    Returns:
        Tuple containing the softmax output and the input vector itself, saved for backpropagation purposes.
    """
    cache = z
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z), axis=1))
    return sm, cache


def relu(z):
    """
    Compute the ReLU activation function for a given input vector.

    Args:
        z (numpy.ndarray): Input vector.

    Returns:
        Tuple containing the ReLU output and the input vector itself, saved for backpropagation purposes.
    """
    s = np.maximum(0, z)
    cache = z
    return s, cache


def softmax_backward(dA, cache):
    """
    Compute the derivative of the softmax activation function.

    Args:
        dA (numpy.ndarray): Gradient of the loss function with respect to the softmax output.
        cache (numpy.ndarray): Input vector.

    Returns:
        Gradient of the loss function with respect to the input vector.
    """
    z = cache
    z -= np.max(z)
    s = (np.exp(z).T / np.sum(np.exp(z), axis=1))
    dZ = dA * s * (1 - s)
    return dZ


def relu_backward(dA, cache):
    """
    Compute the derivative of the ReLU activation function.

    Args:
        dA (numpy.ndarray): Gradient of the loss function with respect to the ReLU output.
        cache (numpy.ndarray): Input vector.

    Returns:
        Gradient of the loss function with respect to the input vector.
    """
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
    dZ[Z <= 0] = 0
    return dZ