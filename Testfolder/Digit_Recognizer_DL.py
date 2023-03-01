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

# This function implements the softmax activation function.
def softmax(z):
    cache = z
    z -= np.max(z)  # Normalize to avoid numerical instability
    sm = (np.exp(z).T / np.sum(np.exp(z), axis=1))  # Computes softmax activation values
    return sm, cache


# This function implements the Rectified Linear Unit (ReLU) activation function.
def relu(z):
    s = np.maximum(0, z)  # Computes ReLU activation values by taking the maximum between 0 and z
    cache = z
    return s, cache


# This function implements the backward propagation for the softmax activation function.
def softmax_backward(dA, cache):
    z = cache
    z -= np.max(z)  # Normalize to avoid numerical instability
    s = (np.exp(z).T / np.sum(np.exp(z), axis=1))  # Computes softmax activation values
    dZ = dA * s * (1 - s)  # Computes derivative of the loss function w.r.t. the inputs
    return dZ


# This function implements the backward propagation for the ReLU activation function.
def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)  # Converts dA to correct object type
    dZ[Z <= 0] = 0  # Computes derivative of the loss function w.r.t. the inputs
    return dZ


# The following code is not part of any function. It's just an example of how to use the activation functions.

# Example usage of the softmax function
sm, cache = softmax(np.array([[1, 2], [3, 4], [5, 6]]))
print("Softmax output:", sm)

# Example usage of the ReLU function
s, cache = relu(np.array([[1, -2], [-3, 4], [5, -6]]))
print("ReLU output:", s)

# Example usage of softmax_backward function
dZ = softmax_backward(np.array([[1, 2], [-1, 1], [1, -1]]), cache)
print("softmax_backward output:", dZ)

# Example usage of relu_backward function
dZ = relu_backward(np.array([[1, 2], [-1, 1], [1, -1]]), cache)
print("relu_backward output:", dZ)