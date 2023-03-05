# This file contains functions for computing the softmax and ReLU activation functions,
# along with their derivatives for use in neural networks.

# The softmax function takes in an array of values and returns a probability distribution
# over those values. It first subtracts the max value from the array to avoid numerical
# instability, then exponentiates each value and divides by the sum of all exponentiated values.
# It also saves the original array for use in backpropagation.

# The relu function takes in an array of values and returns the elementwise maximum of 0 and
# the input array. It also saves the original array for use in backpropagation.

# The softmax_backward function computes the derivative of the softmax function with
# respect to its input, given the derivative with respect to its output and the original input.

# The relu_backward function computes the derivative of the relu function with
# respect to its input, given the derivative with respect to its output and the original input.