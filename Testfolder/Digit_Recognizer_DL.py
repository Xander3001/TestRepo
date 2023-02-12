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



# # import numpy and matplotlib
# import numpy as np
# import matplotlib.pyplot as plt
# 
# # Implementing Relu function
# 
# 
# def softmax(z):
#     # Keeping only negative values
#     cache = z
#     z -= np.max(z)
#     # Rounding to 16 decimal points
#     sm = (np.exp(z).T / np.sum(np.exp(z), axis=1))
#     return sm, cache
# 
# 
# def relu(z):
# 
#     s = np.maximum(0, z)
#     # Keeping only negative values
#     cache = z
#     return s, cache
# 
# 
# def softmax_backward(dA, cache):
# 
#     # Keeping only negative values
#     z = cache
#     z -= np.max(z)
#     s = (np.exp(z).T / np.sum(np.exp(z), axis=1))
#     dZ = dA * s * (1 - s)
#     return dZ
# 
# 
# def relu_backward(dA, cache):
#     # Keep only positive zeros 
#     Z = cache
#     dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
#     dZ[Z <= 0] = 0
#     return dZ