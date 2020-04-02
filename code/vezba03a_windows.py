#!/usr/bin/env python3
#
# Copyright by Branislav Gerazov 2016 - 2020
#
# See the file LICENSE for the license associated with this software.
#
# Author(s):
#   Branislav Gerazov, March 2016 - 2020

"""
Digital Audio Processing

Excercise 03: Windows.

@author: Branislav Gerazov
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import fftpack as fp
from scipy import signal as sig

# %% plot window and its spectrum
# Hamming
m = 512  # analysis range
n = 64  # window length
mh = m // 2
nh = n // 2

win = np.zeros(m)
win[mh - nh: mh + nh] = sig.get_window('hamming', n)
win_spec = fp.fft(win, m)
win_spec = np.abs(win_spec) / n
eps = 1e-8
win_spec[win_spec < eps] = eps
win_spec = 20 * np.log10(win_spec)
win_spec = fp.fftshift(win_spec)

# %% plot
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(win)
plt.axis([0, m, 0, 1.1])  # [xmin, xmax, ymin, ymax]
plt.grid()
plt.subplot(122)
plt.plot(win_spec)
plt.grid()
plt.axis([0, m, -100, 10])
