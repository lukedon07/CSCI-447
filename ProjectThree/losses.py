import numpy as np

EPS = 1e-12


def cross_entropy(Y, P):
    """Mean over samples of -sum_k y_k log p_k.  Y is one-hot, P are probabilities."""
    return float(-np.mean(np.sum(Y * np.log(np.clip(P, EPS, 1.0)), axis=1)))


def mse(Y, Yhat):
    """Mean of squared errors over all samples and output columns."""
    return float(np.mean((Yhat - Y) ** 2))


def softmax_cross_entropy_delta(Y, P):
    return (P - Y) / Y.shape[0]


def linear_mse_delta(Y, Yhat):
    return 2.0 * (Yhat - Y) / Y.size