import numpy as np
import matplotlib.pyplot as plt

def softmax(z):
    """
    Implementation of the softmax activation function.

    Args:
    z: numpy array of any shape, input to activation function.

    Returns:
    softmax activation, which keeps the shape of the input.
    cache: returns the input, which will be used in later calculations.
    """
    cache = z
    z -= np.max(z) # Normalize the input array to avoid any overflows while computing the exponentials 
    sm = (np.exp(z).T / np.sum(np.exp(z), axis=1))
    return sm, cache

def relu(z):
    """
    Implementation of the ReLU (Rectified Linear Unit) activation function.

    Args:
    z: numpy array of any shape, input to activation function.

    Returns:
    ReLU activation (max(0,x)), which keeps the shape of the input.
    cache: returns the input, which will be used in later calculations.
    """
    s = np.maximum(0, z)
    cache = z
    return s, cache

def softmax_backward(dA, cache):
    """
    Implementation of the derivative of softmax activation function's backward propagation. 

    Args:
    dA: post-activation gradient numpy array of any shape
    cache: cache returned by softmax() function


    Returns:
    Derivative of the softmax activation function for backward propagation.
    """
    z = cache
    z -= np.max(z) # Normalize the input array to avoid any overflows while computing the exponentials 
    s = (np.exp(z).T / np.sum(np.exp(z), axis=1))
    dZ = dA * s * (1 - s) # Computes the derivative of the softmax function  
    return dZ

def relu_backward(dA, cache):
    """
    Implementation of the derivative of ReLU activation function's backward propagation.

    Args:
    dA: post-activation gradient numpy array of any shape
    cache: cache returned by relu() function

    Returns:
    Derivative of the ReLU activation function for backward propagation.
    """
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dZ to a correct object.
    dZ[Z <= 0] = 0  # Backpropagation through ReLU only when x > 0
    return dZ