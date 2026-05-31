# src/extra_experiments/exp1_sparse_poly/data.py

import numpy as np
from .config import *

def generate_data():
    rng = np.random.default_rng(SEED)

    d = D
    n_train = N_TRAIN
    n_test = N_TEST

    S_size = int(d * ACTIVE_RATIO)
    S = np.arange(S_size)

    X_train = rng.normal(size=(n_train, d))
    X_test = rng.normal(size=(n_test, d))

    a = rng.normal(size=S_size)
    b = rng.normal(scale=0.3, size=(S_size, S_size))
    c = rng.normal(scale=0.1, size=(S_size, S_size, S_size))

    def f(X):
        Xs = X[:, S]
        y = (
            np.sum(a * Xs, axis=1)
            + np.sum(b * (Xs[:, :, None] * Xs[:, None, :]), axis=(1,2))
            + np.sum(
                c * (Xs[:, :, None, None] * Xs[:, None, :, None] * Xs[:, None, None, :]),
                axis=(1,2,3)
            )
        )
        return y

    y_train = f(X_train) + rng.normal(scale=NOISE_STD, size=n_train)
    y_test = f(X_test) + rng.normal(scale=NOISE_STD, size=n_test)

    return  X_train, y_train, X_test, y_test, S, f

class GroundTruthModel:
    def __init__(self, f):
        self.f = f

    def predict(self, X):
        return self.f(X)
