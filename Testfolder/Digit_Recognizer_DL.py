# This code imports numpy and matplotlib libraries 
import numpy as np
import matplotlib.pyplot as plt


# This function computes softmax activation function and stores the input data in cache 
def softmax(z):
    cache = z
    # To avoid numerical instability and for more accurate results, we subtract the maximum value of z from each element in z
    z -= np.max(z)
    # Compute the exponential of each element in z and normalize the output by dividing each row with the sum of exponential elements in that row
    sm = (np.exp(z).T / np.sum(np.exp(z), axis=1))
    # Return sm (softmax output) and cache (input data) 
    return sm, cache


# This function computes relu activation function and stores the input data in cache 
def relu(z):
    # Apply the max function and element-wise multiplication with a binary mask, which returns 0 for negative elements and 1 for positive elements
    s = np.maximum(0, z)
    # Save input data in cache 
    cache = z
    # Return the activation output and cache 
    return s, cache


# This function computes the backward pass of softmax activation function 
def softmax_backward(dA, cache):
    # Retrieve the input values from cache 
    z = cache
    # To avoid numerical instability and for more accurate results, we subtract the maximum value of z from each element in z
    z -= np.max(z)
    # Compute the exponential of each element in z and normalize the output by dividing each row with the sum of exponential elements in that row
    s = (np.exp(z).T / np.sum(np.exp(z), axis=1))
    # Compute the derivative of the loss function with respect to z using the derivative of softmax function 
    dZ = dA * s * (1 - s)
    # Return the derivative with respect to z
    return dZ


# This function computes the backward pass of relu activation function 
def relu_backward(dA, cache):
    # Retrieve the input values from cache 
    Z = cache
    # Create a binary mask for elements <= 0, which produces a value of 1 for active nodes and 0 for inactive ones. 
    # This helps to keep the most activated weights in the backward pass.
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    # Return the derivative with respect to z
    return dZ