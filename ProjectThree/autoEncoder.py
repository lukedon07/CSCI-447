"""Autoencoder pre-training for the prediction network.

Pipeline (all on the TRAINING fold only; labels are never used in step 1):

  1. train_autoencoder:        [d, e, d] network (linear output, MSE) learns to
                               reconstruct its own input, with e < d.
  2. build_pretrained_network: throw away the decoder (the e -> d layer), keep the
                               input -> encoder weights, and attach
                               Hidden 2 -> Prediction. Result: [d, e, h2, k].
  3. Train that network with backprop, either
        freeze_encoder=True   encoder weights stay locked (only the new layers learn)
        freeze_encoder=False  fine-tuning: every layer is updated

`build_autoencoder_network` does steps 1 and 2 (plus the freeze) in one call.
"""
import math

import numpy as np

from network import FeedForwardNetwork

DEFAULT_AE_HYPERPARAMS = {
    "encoder": None,  # None -> ceil(d / 2), the size used for compressibility C
    "hidden2": 8,
    "activation": "tanh",  # used for the encoder and for hidden layer 2
    "ae_lr": 0.01, "ae_momentum": 0.9, "ae_epochs": 100, "ae_batch_size": 32,
}


def train_autoencoder(X, encoder_size, activation="tanh", lr=0.01, momentum=0.9,
                      epochs=100, batch_size=32, seed=None):
    """Train [d, e, d] to reconstruct X. Returns (autoencoder, history)."""
    X = np.asarray(X, dtype=float)
    d = X.shape[1]
    if not 1 <= encoder_size < d:
        raise ValueError(f"Encoder size must satisfy 1 <= e < d (got e={encoder_size}, d={d})")
    ae = FeedForwardNetwork([d, encoder_size, d], activation, "linear", seed=seed)
    history = ae.fit(X, X, epochs=epochs, lr=lr, batch_size=batch_size, momentum=momentum)
    return ae, history


def build_pretrained_network(ae, hidden2, n_out, task, seed=None):
    """Keep the autoencoder's encoder layer, drop its decoder, add Hidden 2 and Prediction."""
    d, e = ae.layer_sizes[0], ae.layer_sizes[1]
    output = "softmax" if task == "classification" else "linear"
    net = FeedForwardNetwork([d, e, hidden2, n_out], ae.hidden_activation, output, seed=seed)
    net.W[0] = ae.W[0].copy()  # encoder weights and biases (copies, not views)
    net.b[0] = ae.b[0].copy()
    return net


def build_autoencoder_network(Xtr, n_out, task, hp, seed, freeze_encoder=False):
    """Pretrain on Xtr, then return the ready-to-train prediction network.

    The pretrained autoencoder is kept on the result as `net.autoencoder`
    so reconstruction error can be reported.
    """
    if Xtr is None:
        raise ValueError("Autoencoder models need the training features Xtr")
    hp = {**DEFAULT_AE_HYPERPARAMS, **hp}
    d = Xtr.shape[1]
    e = hp["encoder"] if hp["encoder"] is not None else math.ceil(d / 2)

    ae, history = train_autoencoder(Xtr, e, hp["activation"], hp["ae_lr"], hp["ae_momentum"],
                                    hp["ae_epochs"], hp["ae_batch_size"], seed=seed)
    net = build_pretrained_network(ae, hp["hidden2"], n_out, task, seed=seed + 10_000)
    if freeze_encoder:
        net.freeze([0])
    net.autoencoder = ae
    net.pretrain_history = history
    return net