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

Excercise 07: Perceptron.

@author: Branislav Gerazov
"""
import numpy as np
from matplotlib import pyplot as plt


# %% class neuron
class Neuron:
    def __init__(self, weights=None, bias=None, n_input=None):
        if weights is None:
            self.weights = np.random.normal(size=n_input)
            self.bias = np.random.normal()
        else:
            self.weights = weights
            self.bias = bias

    def feedforward(self, x):
        a = np.sum(self.weights * x) + self.bias
        y = 1 if a > 0 else 0  # classification
        # y = a  # regression
        return y

    def update(self):
        self.weights -= self.grad_weights * self.learn
        self.bias -= self.grad_bias * self.learn

    def backprop(self, loss, x):
        self.grad_weights = -2*loss*x
        self.grad_bias = -2*loss

    def loss(self, y, y_pred):
        return y - y_pred

    def train(self, xs, ys, epochs=20, learn=0.01):
        self.learn = learn
        for epoch in range(epochs):
            print(epoch, end='')
            epoch_loss = 0
            for x, y in zip(xs, ys):
                y_pred = self.feedforward(x)
                loss = self.loss(y, y_pred)
                self.backprop(loss, x)
                self.update()
                epoch_loss += loss ** 2
            print('->', epoch_loss / xs.size)


# %% test classification
xs = np.array([[0, 0],
               [0, 1],
               [1, 0],
               [1, 1]])
neuron = Neuron(
        weights=np.array([1, 1]), bias=-1,  # AND
        # weights=np.array([-1, -1]), bias=1.5,  # NAND
        )
for x in xs:
    print(x, '->', neuron.feedforward(x))

# %% plot
ys = np.array([0, 0, 0, 1])
ys = np.array([1, 1, 1, 0])
plt.figure()
plt.scatter(xs[:, 0], xs[:, 1], c=['r', 'r', 'r', 'b'])
x_axis = np.array([-.1, 1.1])
dec_bound = (-neuron.weights[0]*x_axis - neuron.bias) / neuron.weights[1]
plt.plot(x_axis, dec_bound, 'g', lw=2)
plt.grid()

# %% train neuron
# ys = np.array([0, 0, 0, 1])  # AND
ys = np.array([1, 1, 1, 0])  # NAND
neuron = Neuron()
neuron.train(xs, ys)
print(neuron.weights, neuron.bias)

# %% interactive training
plt.figure()
for epoch in range(20):
    np.random.seed(42)
    neuron = Neuron(n_input=2)
    neuron.train(xs, ys, epochs=epoch, learn=0.02)
    print(neuron.weights, neuron.bias)
    plt.cla()
    plt.scatter(xs[:, 0], xs[:, 1], c=['r', 'r', 'r', 'b'])
    dec_bound = (-neuron.weights[0]*x_axis - neuron.bias) / neuron.weights[1]
    plt.plot(x_axis, dec_bound, 'g', lw=2)
    plt.grid()
    plt.axis([-0.1, 1.1, -0.1, 3.1])
    plt.pause(0.2)

# %% data regression
# in Neuron change feedforward for regression!

# generate data
np.random.seed(42)
xs = np.random.rand(10) * 2*np.pi
ys = np.sin(xs)

# train neuron
neuron = Neuron(n_input=1)
neuron.train(xs, ys)

# predict
x_axis = np.linspace(0, 2*np.pi, 100)
y_pred = [neuron.feedforward(x) for x in x_axis]

plt.figure()
plt.scatter(xs, ys)
plt.plot(x_axis, y_pred, 'g', lw=2)
plt.grid()

# %% interactive training
plt.figure()
for epoch in range(1, 51):
    np.random.seed(42)
    neuron = Neuron(n_input=1)
    neuron.train(xs, ys, epochs=epoch, learn=0.01)
    y_pred = [neuron.feedforward(x) for x in x_axis]
    plt.cla()
    plt.scatter(xs, ys)
    plt.plot(x_axis, y_pred, 'g', lw=2)
    plt.grid()
    plt.axis([0, 6.3, -1.4, 1.1])
    plt.pause(0.1)
