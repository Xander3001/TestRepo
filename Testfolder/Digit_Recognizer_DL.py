"""
This script defines two activation functions: softmax and relu, and their respective
backward functions softmax_backward and relu_backward. These functions can be used in
neural network architectures.

Functions:
- softmax: computes the softmax of a given input.
- relu: computes the rectified linear unit (ReLU) activation function of a given input.
- softmax_backward: computes the backward pass of the softmax function.
- relu_backward: computes the backward pass of the ReLU function.

Inputs:
- z: an array representing input data.

Returns:
- For softmax:
    - sm: an array of same shape as input z, containing softmax values.
    - cache: an array representing input data.
- For relu:
    - s: an array of same shape as input z, containing ReLU values.
    - cache: an array representing input data.
- For softmax_backward:
    - dZ: an array of same shape as input dA, containing the gradient of the loss with
    respect to z.
    - cache: an array representing input data.
- For relu_backward:
    - dZ: an array of same shape as input dA, containing the gradient of the loss with
    respect to z.
    - cache: an array representing input data.
"""

import numpy as np
import matplotlib.pyplot as plt


def softmax(z):
    """
    Compute the softmax of a given input.

    Inputs:
    - z: an array representing input data.

    Returns:
    - sm: an array of same shape as input z, containing softmax values.
    - cache: an array representing input data.
    """

    cache = z
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z), axis=1))
    return sm, cache


def relu(z):
    """
    Compute the rectified linear unit (ReLU) activation function of a given input.

    Inputs:
    - z: an array representing input data.

    Returns:
    - s: an array of same shape as input z, containing ReLU values.
    - cache: an array representing input data.
    """

    s = np.maximum(0, z)
    cache = z
    return s, cache


def softmax_backward(dA, cache):
    """
    Compute the backward pass of the softmax function.

    Inputs:
    - dA: an array representing the gradient of the loss with respect to output of the
    softmax function.
    - cache: an array representing input data.

    Returns:
    - dZ: an array of same shape as input dA, containing the gradient of the loss with
    respect to z.
    """

    z = cache
    z -= np.max(z)
    s = (np.exp(z).T / np.sum(np.exp(z), axis=1))
    dZ = dA * s * (1 - s)
    return dZ


def relu_backward(dA, cache):
    """
    Compute the backward pass of the ReLU function.

    Inputs:
    - dA: an array representing the gradient of the loss with respect to output of the
    ReLU function.
    - cache: an array representing input data.

    Returns:
    - dZ: an array of same shape as input dA, containing the gradient of the loss with
    respect to z.
    """

    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
    dZ[Z <= 0] = 0
    return dZ