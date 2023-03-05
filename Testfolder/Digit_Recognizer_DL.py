# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt


# This function returns the softmax activation of input 'z' and stores the original input in 'cache'
def softmax(z):
    cache = z
    z -= np.max(z)  # subtracting maximum value of 'z' for numerical stability
    sm = (np.exp(z).T / np.sum(np.exp(z), axis=1))  # softmax activation applied to 'z'
    return sm, cache


# This function returns the relu activation of input 'z' and stores the original input in 'cache'
def relu(z):
    s = np.maximum(0, z)  # relu activation applied to 'z'
    cache = z
    return s, cache


# This function returns the derivative of softmax activation given the output dA and original input 'cache'
def softmax_backward(dA, cache):
    z = cache
    z -= np.max(z)  # subtracting maximum value of 'z' for numerical stability
    s = (np.exp(z).T / np.sum(np.exp(z), axis=1))  # softmax activation applied to 'z'
    dZ = dA * s * (1 - s)  # element-wise multiplication of dA and derivative of softmax activation
    return dZ


# This function returns the derivative of relu activation given the output dA and original input 'cache'
def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)  # creating a copy of dA
    dZ[Z <= 0] = 0  # applying derivative of relu activation to dZ
    return dZ