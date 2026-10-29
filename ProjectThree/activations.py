import numpy as np


def sigmoid(z):
    z = np.clip(z, -500, 500)  # avoid overflow in exp
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_deriv(a):
    return a * (1.0 - a)


def tanh(z):
    return np.tanh(z)


def tanh_deriv(a):
    return 1.0 - a ** 2


def softmax(z):
    """Row-wise softmax. Subtracting the row max keeps exp() from overflowing."""
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def identity(z):
    return z


# name -> (function, derivative in terms of the output)
HIDDEN_ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_deriv),
    "tanh": (tanh, tanh_deriv),
}