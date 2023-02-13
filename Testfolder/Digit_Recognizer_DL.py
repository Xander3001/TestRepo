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



# import numpy as np
# import matplotlib.pyplot as plt
# 
# # softmax function sigmoid(x)=1/1+e^(-x)
# 
# def softmax(z):
#     cache = z
#     z -= np.max(z)
#     sm = (np.exp(z).T / np.sum(np.exp(z), axis=1))
#     return sm, cache
# 
# 
# # sigmoid function sigmoid(x)=1/1+e^(-x)
# def sigmoid(z):
# 
#     s = np.maximum(0, z)
#     cache = z
#     return s, cache
# 
# 
# # relu backward function
# def softmax_backward(dA, cache):
# 
#     z = cache
#     z -= np.max(z)
#     s = (np.exp(z).T / np.sum(np.exp(z), axis=1))
#     dZ = dA * s * (1 - s)
#     return dZ
# # sigmoid backward function
# 
# 
# def relu_backward(dA, cache):
#  
#     Z = cache
#     dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
#     dZ[Z <= 0] = 0
#     return dZ