import numpy as np
import matplotlib.pyplot as plt


def softmax(z):
    cache = z
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z), axis=1))
    return sm, cache


def relu(z):

    s = np.maximum(0, z)
    cache = z
    return s, cache


def softmax_backward(dA, cache):

    z = cache
    z -= np.max(z)
    s = (np.exp(z).T / np.sum(np.exp(z), axis=1))
    dZ = dA * s * (1 - s)
    return dZ


def relu_backward(dA, cache):
 
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
    dZ[Z <= 0] = 0
    return dZ



import numpy as np
import matplotlib.pyplot as plt


def softmax(z):
    """
    Applies the softmax function to the input matrix.

    Args:
    z: A numpy array containing the input matrix.

    Returns:
    A numpy array that contains the softmax of the input matrix.
    """
    cache = z
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z), axis=1))
    return sm, cache


def relu(z):
    """
    Applies the rectified linear unit activation function to the input matrix.

    Args:
    z: A numpy array containing the input matrix.

    Returns:
    A numpy array that contains the output of the activation function.
    """
    s = np.maximum(0, z)
    cache = z
    return s, cache


def softmax_backward(dA, cache):
    """
    Computes the gradient of the softmax activation function.

    Args:
    dA: A numpy array containing the derivative of the cost function with respect to the output of the softmax function.
    cache: A numpy array containing the input matrix.

    Returns:
    A numpy array that contains the derivative of the cost function with respect to the input matrix.
    """
    z = cache
    z -= np.max(z)
    s = (np.exp(z).T / np.sum(np.exp(z), axis=1))
    dZ = dA * s * (1 - s)
    return dZ


def relu_backward(dA, cache):
    """
    Computes the gradient of the rectified linear unit activation function.

    Args:
    dA: A numpy array containing the derivative of the cost function with respect to the output of the activation function.
    cache: A numpy array containing the input matrix.

    Returns:
    A numpy array containing the derivative of the cost function with respect to the input matrix.
    """
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
    dZ[Z <= 0] = 0
    return dZ


# Example usage:
# x = np.array([[1, 2, 3], [4, 5, 6]])
# sm, sm_cache = softmax(x)
# relu_out, relu_cache = relu(x)
# sm_backward = softmax_backward(sm, sm_cache)
# relu_backward = relu_backward(relu_out, relu_cache)