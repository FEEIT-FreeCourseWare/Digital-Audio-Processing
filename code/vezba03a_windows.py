#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2016 by Branislav Gerazov
#
# See the file LICEnSE for the license associated with this software.
#
# Author(s):
#   Branislav Gerazov, March 2016

"""
Digital Audio Systems

Excercise 03: Windows.

@author: Branislav Gerazov
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import fftpack as fft
from scipy import signal as sig

# %% plot window and its spectrum
# Hamming
m = 512  # analysis range
n = 64  # window length
Mh = m/2
nh = n/2

win = np.zeros(m)
win[Mh-nh:Mh+nh] = sig.get_window('hamming', n)
win_spec = fft.fft(win, m)
win_spec = np.abs(win_spec) / n
eps = np.finfo(float).eps
win_spec[win_spec < eps] = eps
win_spec = 20 * np.log10(win_spec)
win_spec = fft.fftshift(win_spec)

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
