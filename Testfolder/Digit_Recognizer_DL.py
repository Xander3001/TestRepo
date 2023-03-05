# importing the required libraries
import numpy as np
import matplotlib.pyplot as plt


def softmax(z):
    '''
    This function returns the softmax activation function for a given input z.
    
    Parameters:
    z (numpy array): input array
    
    Returns:
    sm (numpy array): softmax function applied to the input array
    cache (numpy array): input array 'z' is cached to be used during backpropagation
    '''
    
    # cache the input array
    cache = z
    
    # applying the softmax function
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z), axis=1))
    
    return sm, cache


def relu(z):
    '''
    This function returns the relu activation function for a given input z.
    
    Parameters:
    z (numpy array): input array
    
    Returns:
    s (numpy array): relu function applied to the input array
    cache (numpy array): input array 'z' is cached to be used during backpropagation
    '''
    
    # applying the relu function
    s = np.maximum(0, z)
    
    # cache the input array
    cache = z
    
    return s, cache


def softmax_backward(dA, cache):
    '''
    This function returns the derivative of the softmax function for a given input 'dA'.
    
    Parameters:
    dA (numpy array): derivative of the cost function with respect to the activation output
    cache (numpy array): input array, cached during forward propagation
    
    Returns:
    dZ (numpy array): derivative of the cost function with respect to the input 'z'
    '''
    
    # retrieving the cached input array
    z = cache
    
    # applying the softmax function
    z -= np.max(z)
    s = (np.exp(z).T / np.sum(np.exp(z), axis=1))
    
    # calculating the derivative of the cost function
    dZ = dA * s * (1 - s)
    
    return dZ


def relu_backward(dA, cache):
    '''
    This function returns the derivative of the relu function for a given input 'dA'.
    
    Parameters:
    dA (numpy array): derivative of the cost function with respect to the activation output
    cache (numpy array): input array, cached during forward propagation
    
    Returns:
    dZ (numpy array): derivative of the cost function with respect to the input 'z'
    '''
    
    # retrieving the cached input array
    Z = cache
    
    # calculating the derivative of the cost function
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    
    return dZ