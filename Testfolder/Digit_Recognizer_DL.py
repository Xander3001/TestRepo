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


def softmax(z):
    cache = z
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z), axis=1))
    return sm, cache
 # Thus we convert a value between 0 to 1 in a form that the sum 
    # of the exponentials of the terms equals 1
    # z - max(z) is used for computing softmax, there are other variants
    #   z - sum(z)

def relu(z):

    s = np.maximum(0, z)
    cache = z
    # This function is used for converting a value into a 
    # negative value or 0
    return s, cache

    # If 0 is considered, then the derivative
    # is zero.
# This function is the derivative of the exponentials of the softmax function
def softmax_backward(dA, cache):

    z = cache
    z -= np.max(z)
    s = (np.exp(z).T / np.sum(np.exp(z), axis=1))
    dZ = dA * s * (1 - s)
    # This derivative is used when we have many inputs 
    # from one layer to another.
    return dZ


def relu_backward(dA, cache):
 
    # But when positive, then the derivative is equal to 1.
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
    dZ[Z <= 0] = 0
    return dZ