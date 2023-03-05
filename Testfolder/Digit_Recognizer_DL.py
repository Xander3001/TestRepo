# This code defines four functions: 
# softmax, which calculates the softmax activation function of a given input array z
# relu, which calculates the ReLU (Rectified Linear Unit) activation function of a given input array z
# softmax_backward, which calculates the backward pass for a softmax activation function given the derivative dA and the cache (the input z from the forward pass)
# relu_backward, which calculates the backward pass for a ReLU activation function given the derivative dA and the cache (the input z from the forward pass)

import numpy as np
import matplotlib.pyplot as plt


def softmax(z):
    """
    Calculate softmax activation function for a given input array z.
    
    Arguments:
    z -- numpy array of shape (m, n), where m is the number of samples and n is the number of classes
    
    Returns:
    sm -- numpy array of shape (m, n), representing the softmax activation function of z
    cache -- z, stored for use in backward pass calculations
    """
    cache = z
    z -= np.max(z) # to avoid overflow
    sm = (np.exp(z).T / np.sum(np.exp(z), axis=1)) # formula for softmax activation function
    return sm, cache


def relu(z):
    """
    Calculate ReLU activation function for a given input array z.
    
    Arguments:
    z -- any numpy array of any shape
    
    Returns:
    s -- numpy array of the same shape as z, representing the ReLU activation function of z
    cache -- z, stored for use in backward pass calculations
    """
    s = np.maximum(0, z) # formula for ReLU activation function
    cache = z
    return s, cache


def softmax_backward(dA, cache):
    """
    Calculate the backward pass for a softmax activation function given the derivative dA and the cache (the input z from the forward pass).
    
    Arguments:
    dA -- numpy array of shape (m, n), representing the derivative of the loss with respect to the output of the softmax activation function
    cache -- z, stored from the forward pass
    
    Returns:
    dZ -- numpy array of the same shape as dA, representing the derivative of the loss with respect to the input of the softmax activation function
    """
    z = cache
    z -= np.max(z) # to avoid overflow
    s = (np.exp(z).T / np.sum(np.exp(z), axis=1)) # formula for softmax activation function
    dZ = dA * s * (1 - s) # formula for derivative of softmax activation function
    return dZ


def relu_backward(dA, cache):
    """
    Calculate the backward pass for a ReLU activation function given the derivative dA and the cache (the input z from the forward pass).
    
    Arguments:
    dA -- numpy array of any shape, representing the derivative of the loss with respect to the output of the ReLU activation function
    cache -- z, stored from the forward pass
    
    Returns:
    dZ -- numpy array of the same shape as dA, representing the derivative of the loss with respect to the input of the ReLU activation function
    """
    Z = cache
    dZ = np.array(dA, copy=True) # converting dz to a correct object
    dZ[Z <= 0] = 0 # formula for derivative of ReLU activation function
    return dZ