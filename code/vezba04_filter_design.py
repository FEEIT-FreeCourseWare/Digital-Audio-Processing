#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Copyright 2016 by Branislav Gerazov
#
# See the file LICENSE for the license associated with this software.
#
# Author(s):
#   Branislav Gerazov, Apr 2016

"""
Digital Audio Systems

Excercise 04: Filter Design.

@author: Branislav Gerazov
"""
from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from math import pi
from scipy.io import wavfile
import das
from scipy import fftpack as fp
from scipy import signal as sig

#%% define ideal filter
fs = 44100
w_g = 5000 / (fs/2)  # normalized cut-off frequency
Nfft = fs
f = np.linspace(0,pi,Nfft/2+1)
H_ideal = np.zeros(f.size)
H_ideal[f < w_g * pi] = 1
H_ideal = np.append(H_ideal, H_ideal[-2:0:-1])
h_ideal = fp.ifft(H_ideal, Nfft)
h_ideal = fp.fftshift(h_ideal)

#%% windows
N = 128 + 1
Nh = (N-1)/2
Nffth = Nfft/2
n_win = np.arange(Nffth-Nh,Nffth+Nh+1)

win_long = np.zeros(Nfft)
win_long[tuple(n_win),] = sig.get_window('boxcar', N)
h_rect_long = h_ideal * win_long

win = sig.get_window('boxcar', N)
h_rect = h_ideal[tuple(n_win),] * win

#%% plot designed filters
plt.figure()

plt.subplot(211)
plt.plot(h_ideal,linewidth=1,alpha=1)
plt.plot(n_win,h_rect,linewidth=2,alpha=.8)
#plt.plot(.25*win_long,linewidth=2,alpha=.8,color='g')
plt.grid()
plt.axis([Nffth-1.5*Nh,Nffth+1.5*Nh,-0.1,0.25])

plt.subplot(212)
f, H_ideal = das.get_spectrum(h_ideal,fs)
H_ideal = H_ideal - np.max(H_ideal)
plt.plot(f, H_ideal)

#f, H_rect = das.get_spectrum(h_rect,fs)
f, H_rect = das.get_spectrum(h_rect_long,fs)
H_rect = H_rect - np.max(H_rect)
plt.plot(f, H_rect,linewidth=1.5,alpha=.8)
plt.grid()
plt.axis([0,10000,-90,10])
