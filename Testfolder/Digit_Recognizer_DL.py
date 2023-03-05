# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Defining the softmax function
def softmax(z):
    """
    Arguments:
    z -- A numpy array of shape (n_samples, n_classes) containing the input logits
    
    Returns:
    sm -- A numpy array of shape (n_samples, n_classes) containing the output softmax probabilities
    cache -- A tuple containing the input 'z' which will be used during backpropagation
    """
    cache = z
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z), axis=1))
    return sm, cache

# Defining the ReLU function
def relu(z):
    """
    Arguments:
    z -- A numpy array of shape (n_units_prev_layer, n_samples) containing the input activations
    
    Returns:
    s -- A numpy array of shape (n_units_prev_layer, n_samples) containing the output activations after applying ReLU
    cache -- A tuple containing the input 'z' which will be used during backpropagation
    """
    s = np.maximum(0, z)
    cache = z
    return s, cache

# Defining the softmax_backward function
def softmax_backward(dA, cache):
    """
    Arguments:
    dA -- A numpy array of shape (n_samples, n_classes) containing the derivative of cost w.r.t. softmax output
    cache -- A tuple containing the input 'z' which was obtained during forward propagation
    
    Returns:
    dZ -- A numpy array of shape (n_samples, n_classes) containing the derivative of cost w.r.t. the input 'z'
    """
    z = cache
    z -= np.max(z)
    s = (np.exp(z).T / np.sum(np.exp(z), axis=1))
    dZ = dA * s * (1 - s)
    return dZ

# Defining the relu_backward function
def relu_backward(dA, cache):
    """
    Arguments:
    dA -- A numpy array of shape (n_units, n_samples) containing the derivative of cost w.r.t. the output activations
    cache -- A numpy array of shape (n_units, n_samples) containing the input activations obtained during forward prop
    
    Returns:
    dZ -- A numpy array of shape (n_units, n_samples) containing the derivative of cost w.r.t. the input 'z'
    """
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to correct object type
    dZ[Z <= 0] = 0
    return dZ

# Visualizing the sigmoid and ReLU functions
x = np.linspace(-10, 10, 100)
plt.plot(x, softmax(np.array([x, 1-x]).T)[0], label='Softmax')
plt.plot(x, relu(x)[0], label='ReLU')
plt.legend()
plt.show()