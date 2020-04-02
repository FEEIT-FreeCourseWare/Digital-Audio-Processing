#!/usr/bin/env python3
#
# Copyright by Branislav Gerazov 2019 - 2020
#
# See the file LICENSE for the license associated with this software.
#
# Author(s):
#   Branislav Gerazov, May 2019 - 2020

"""
Digital Audio Processing

Excercise 07: Toy regression with scikit-learn.

@author: Branislav Gerazov
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn import neural_network

# %% generate data
np.random.seed(23)
xs = np.random.rand(10) * 2*np.pi
xs = xs[:, np.newaxis]  # for n_feats
ys = np.sin(xs)

# %% init regressor
reg = neural_network.MLPRegressor(
        hidden_layer_sizes=(5),
        activation='tanh',
        alpha=0,
        learning_rate_init=0.01,
        max_iter=3000,
        tol=1e-6,
        early_stopping=False,
        validation_fraction=0.2,
        random_state=42,
        verbose=1,
        )

# %% train - fit
reg.fit(xs, ys)

# %% predict
x_axis = np.linspace(-.1, 7, 100)
x_axis = x_axis[:, np.newaxis]
y_pred = reg.predict(x_axis)

# %% plot results
plt.figure()
plt.scatter(xs, ys, c='k')
plt.plot(x_axis, y_pred)
plt.grid()
plt.tight_layout()

# %% tweak model power
np.random.seed(0)
ys = np.sin(xs) + np.random.normal(size=xs.shape) * 0.2
# ys = np.sin(xs)
plt.figure()
plt.scatter(xs, ys, c='k')
for neuron in [1, 3, 4, 5]:
    reg = neural_network.MLPRegressor(
        hidden_layer_sizes=(neuron),
        activation='tanh',
        alpha=0,
        learning_rate_init=0.01,
        max_iter=3000,
        tol=1e-9,
        random_state=0,
        verbose=0,
        )
    reg.fit(xs, ys.ravel())
    y_pred = reg.predict(x_axis)
    plt.plot(x_axis, y_pred, lw=3, alpha=0.5, label=f'{neuron} neurons')

plt.grid()
plt.legend()
plt.tight_layout()

# %% tweak regularisation
np.random.seed(0)
ys = np.sin(xs) + np.random.normal(size=xs.shape) * 0.2
plt.figure()
plt.scatter(xs, ys, c='k')
for alpha in [0, 0.01, 0.1, 1]:
    reg = neural_network.MLPRegressor(
        hidden_layer_sizes=(5),
        activation='tanh',
        alpha=alpha,
        learning_rate_init=0.01,
        max_iter=3000,
        tol=1e-9,
        random_state=0,
        verbose=0,
        )
    reg.fit(xs, ys.ravel())
    y_pred = reg.predict(x_axis)
    plt.plot(x_axis, y_pred, lw=3, alpha=0.5, label=f'alpha {alpha:.3f}')

plt.grid()
plt.legend()
plt.tight_layout()
