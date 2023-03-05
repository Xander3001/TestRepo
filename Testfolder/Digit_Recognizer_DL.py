# The code contains four functions that are used in a neural network:
# - softmax: applies the softmax function element-wise to the input array z and returns the result and the cache (input array)
# - relu: applies the ReLU (rectified linear unit) activation function element-wise to the input array z and returns the result and the cache (input array)
# - softmax_backward: computes the derivative of the softmax function with respect to the input array using the cached input array and the input derivative dA
# - relu_backward: computes the derivative of the ReLU function with respect to the input array using the cached input array and the input derivative dA

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Defining softmax function
def softmax(z):
    # Caching the input array
    cache = z
    # Normalizing z by subtracting the maximum value of z to prevent overflow
    z -= np.max(z)
    # Applying softmax function and transposing the result to match the shape of input array
    sm = (np.exp(z).T / np.sum(np.exp(z), axis=1))
    # Returning the softmax result and the cache
    return sm, cache

# Defining ReLU function
def relu(z):
    # Applying ReLU function element-wise to the input array z
    s = np.maximum(0, z)
    # Caching the input array
    cache = z
    # Returning the result and cache
    return s, cache

# Defining softmax_backward function
def softmax_backward(dA, cache):
    # Retrieving the cached input array
    z = cache
    # Normalizing z by subtracting the maximum value of z to prevent overflow
    z -= np.max(z)
    # Computing softmax derivative using the input derivative dA, the softmax result, and (1 - softmax result)
    s = (np.exp(z).T / np.sum(np.exp(z), axis=1))
    dZ = dA * s * (1 - s)
    # Returning the derivative
    return dZ

# Defining relu_backward function
def relu_backward(dA, cache):
    # Retrieving the cached input array
    Z = cache
    # Copying the input derivative dA to a new array dZ
    dZ = np.array(dA, copy=True)
    # Setting the elements of dZ where the corresponding elements of Z are less than or equal to 0 to 0 (derivative of ReLU function)
    dZ[Z <= 0] = 0
    # Returning the derivative
    return dZ