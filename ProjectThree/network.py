"""A small feedforward neural network trained with backpropagation.

    layer_sizes = [n_inputs, hidden_1, ..., hidden_k, n_outputs]

* [d, k]            -> no hidden layers: logistic regression (softmax output)
                       or linear regression (linear output)
* [d, h1, h2, k]    -> the two-hidden-layer network
* [d, e, d]         -> an autoencoder (linear output, MSE reconstruction)

Hidden activation: "sigmoid" or "tanh".   Output: "softmax" or "linear".
Loss: cross-entropy for softmax, mean squared error for linear.

Weight layer i connects layer i to layer i+1, so a network with sizes
[d, h1, h2, k] has weight layers 0, 1, 2 (layer 2 is the output layer).
"""
import numpy as np

import activations as act
import losses


class FeedForwardNetwork:
    def __init__(self, layer_sizes, hidden_activation="sigmoid", output="softmax", seed=None):
        if len(layer_sizes) < 2 or any(int(s) < 1 for s in layer_sizes):
            raise ValueError("layer_sizes needs at least [inputs, outputs], all positive")
        if hidden_activation not in act.HIDDEN_ACTIVATIONS:
            raise ValueError("hidden_activation must be 'sigmoid' or 'tanh'")
        if output not in ("softmax", "linear"):
            raise ValueError("output must be 'softmax' or 'linear'")

        self.layer_sizes = [int(s) for s in layer_sizes]
        self.hidden_activation = hidden_activation
        self.output = output
        self._hidden_fn, self._hidden_deriv = act.HIDDEN_ACTIVATIONS[hidden_activation]
        self.rng = np.random.default_rng(seed)

        # Glorot/Xavier uniform initialisation; biases start at zero.
        self.W, self.b = [], []
        for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            limit = np.sqrt(6.0 / (n_in + n_out))
            self.W.append(self.rng.uniform(-limit, limit, size=(n_in, n_out)))
            self.b.append(np.zeros((1, n_out)))
        self.frozen = set()

    @property
    def n_layers(self):
        return len(self.W)

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------
    def forward(self, X):
        """Return the network output; caches every layer's activations."""
        a = np.asarray(X, dtype=float)
        self._acts = [a]
        for i in range(self.n_layers):
            z = a @ self.W[i] + self.b[i]
            if i < self.n_layers - 1:
                a = self._hidden_fn(z)
            else:
                a = act.softmax(z) if self.output == "softmax" else act.identity(z)
            self._acts.append(a)
        return a

    def loss(self, X, Y):
        out = self.forward(X)
        return losses.cross_entropy(Y, out) if self.output == "softmax" else losses.mse(Y, out)

    def compute_gradients(self, X, Y):
        """Backpropagation. Returns (dW, db): gradients of the loss for every layer."""
        Y = np.asarray(Y, dtype=float)
        out = self.forward(X)
        if self.output == "softmax":
            delta = losses.softmax_cross_entropy_delta(Y, out)
        else:
            delta = losses.linear_mse_delta(Y, out)

        dW = [None] * self.n_layers
        db = [None] * self.n_layers
        for i in reversed(range(self.n_layers)):
            dW[i] = self._acts[i].T @ delta
            db[i] = delta.sum(axis=0, keepdims=True)
            if i > 0:  # push the error back through W[i] and the hidden activation
                delta = (delta @ self.W[i].T) * self._hidden_deriv(self._acts[i])
        return dW, db

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X, Y, epochs=100, lr=0.1, batch_size=32, momentum=0.0,
            X_val=None, Y_val=None):
        """Mini-batch gradient descent with optional momentum.

        batch_size=None uses the whole training set each step.
        Returns {"train_loss": [...], "val_loss": [...]} with one entry per epoch.
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of rows")
        if X.shape[1] != self.layer_sizes[0] or Y.shape[1] != self.layer_sizes[-1]:
            raise ValueError(f"Expected X with {self.layer_sizes[0]} columns and "
                             f"Y with {self.layer_sizes[-1]} columns")
        n = X.shape[0]
        batch_size = n if batch_size is None else min(int(batch_size), n)

        vW = [np.zeros_like(w) for w in self.W]
        vb = [np.zeros_like(b) for b in self.b]
        history = {"train_loss": [], "val_loss": []}

        for _ in range(epochs):
            order = self.rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                dW, db = self.compute_gradients(X[idx], Y[idx])
                for i in range(self.n_layers):
                    if i in self.frozen:
                        continue
                    vW[i] = momentum * vW[i] - lr * dW[i]
                    vb[i] = momentum * vb[i] - lr * db[i]
                    self.W[i] += vW[i]
                    self.b[i] += vb[i]
            history["train_loss"].append(self.loss(X, Y))
            if X_val is not None:
                history["val_loss"].append(self.loss(X_val, Y_val))
        return history

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, X):
        """Softmax probabilities (classification) or linear outputs (regression)."""
        return self.forward(X)

    def predict_labels(self, X):
        return self.forward(X).argmax(axis=1)

    # ------------------------------------------------------------------
    # Layer freezing and weight copying (used by the autoencoder pipeline)
    # ------------------------------------------------------------------
    def freeze(self, layers):
        """Stop updating the given weight layers (list of indices)."""
        for i in layers:
            if not 0 <= i < self.n_layers:
                raise IndexError(f"Layer {i} does not exist (0..{self.n_layers - 1})")
        self.frozen |= set(layers)

    def unfreeze(self):
        self.frozen = set()

    def get_weights(self):
        return [(w.copy(), b.copy()) for w, b in zip(self.W, self.b)]

    def set_weights(self, weights):
        if len(weights) != self.n_layers:
            raise ValueError("Wrong number of layers")
        for i, (w, b) in enumerate(weights):
            if w.shape != self.W[i].shape or b.shape != self.b[i].shape:
                raise ValueError(f"Shape mismatch in layer {i}")
        self.W = [np.array(w, dtype=float) for w, _ in weights]
        self.b = [np.array(b, dtype=float) for _, b in weights]
