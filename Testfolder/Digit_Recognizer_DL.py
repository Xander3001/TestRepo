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



Function: softmax

Description: This function computes the softmax activation function for a given input array.

Input:
- z: input array for which softmax activation function should be computed.

Output:
- sm: output array after applying the softmax activation function.
- cache: stores the input array for later use in backpropagation.

Function: relu

Description: This function computes the rectifier linear unit (ReLU) activation function for a given input array.

Input:
- z: input array for which ReLU activation function should be computed.

Output:
- s: output array after applying the ReLU activation function.
- cache: stores the input array for later use in backpropagation.

Function: softmax_backward

Description: This function computes the derivatives of the softmax activation function for a given output array and input array.

Input:
- dA: derivative of the output array with respect to the cost function.
- cache: stores the input array for use in backpropagation.

Output:
- dZ: derivative of the input array with respect to the cost function.

Function: relu_backward

Description: This function computes the derivatives of the ReLU activation function for a given output array and input array.

Input:
- dA: derivative of the output array with respect to the cost function.
- cache: stores the input array for use in backpropagation.

Output:
- dZ: derivative of the input array with respect to the cost function.