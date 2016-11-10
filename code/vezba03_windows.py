#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Copyright 2016 by Branislav Gerazov
#
# See the file LICENSE for the license associated with this software.
#
# Author(s):
#   Branislav Gerazov, March 2016

"""
Digital Audio Systems

Excercise 03: Windows.

@author: Branislav Gerazov
"""
from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from scipy import fftpack as fp
from scipy import signal as sig 

#%% plot window and its spectrum
# Hamming
M = 512  # analysis range
N = 64  # window length 
Mh = M/2
Nh = N/2

w = np.zeros(M)
w[Mh-Nh:Mh+Nh] = sig.get_window('hamming', N)
W = fp.fft(w, M)
Wamp = np.abs(W)/N
eps = np.finfo(float).eps
Wamp[np.where(Wamp < eps)] = eps
Wlog = 20*np.log10(Wamp)
Wlog -= np.max(Wlog)
Wshift = fp.fftshift(Wlog)

#%% plot 
plt.figure(figsize=(12,4))
plt.subplot(121)
plt.plot(w)
plt.axis([0, M, 0,1.1])  #[xmin, xmax, ymin, ymax]
plt.grid()
plt.subplot(122)
plt.plot(Wshift)
plt.grid()
plt.axis([0, M, -100,10])  #[xmin, xmax, ymin, ymax]
