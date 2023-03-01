import numpy as np
import matplotlib.pyplot as plt


def softmax(z):
    """
    Compute the softmax activation function for a given input.

    Parameters:
    z (numpy ndarray): Array of size (num_examples, num_classes) containing the input activations.

    Returns:
    softmax_activation (numpy ndarray): Array of size (num_examples, num_classes) containing the softmax activations.
    cache (numpy ndarray): Array of size (num_examples, num_classes) containing the input `z`.
    """

    cache = z
    z -= np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    sum_exp_z = np.sum(exp_z, axis=1, keepdims=True)
    softmax_activation = exp_z / sum_exp_z

    return softmax_activation, cache


def relu(z):
    """
    Compute the ReLU activation function for a given input.

    Parameters:
    z (numpy ndarray): Array of size (num_examples, num_units) containing the input activations.

    Returns:
    relu_activation (numpy ndarray): Array of size (num_examples, num_units) containing the ReLU activations.
    cache (numpy ndarray): Array of size (num_examples, num_units) containing the input `z`.
    """

    cache = z
    relu_activation = np.maximum(0, z)

    return relu_activation, cache


def softmax_backward(dA, cache):
    """
    Compute the backward pass of the softmax activation function.

    Parameters:
    dA (numpy ndarray): Array of size (num_examples, num_classes) containing the gradient of the loss w.r.t. the softmax activations.
    cache (numpy ndarray): Array of size (num_examples, num_classes) containing the input `z`.

    Returns:
    dZ (numpy ndarray): Array of size (num_examples, num_classes) containing the gradient of the loss w.r.t. the input `z`.
    """

    z = cache
    z -= np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    sum_exp_z = np.sum(exp_z, axis=1, keepdims=True)
    softmax_activation = exp_z / sum_exp_z
    dZ = softmax_activation * (1 - softmax_activation) * dA

    return dZ


def relu_backward(dA, cache):
    """
    Compute the backward pass of the ReLU activation function.

    Parameters:
    dA (numpy ndarray): Array of size (num_examples, num_units) containing the gradient of the loss w.r.t. the ReLU activations.
    cache (numpy ndarray): Array of size (num_examples, num_units) containing the input `z`.

    Returns:
    dZ (numpy ndarray): Array of size (num_examples, num_units) containing the gradient of the loss w.r.t. the input `z`.
    """

    z = cache
    dZ = np.array(dA, copy=True)
    dZ[z <= 0] = 0

    return dZ