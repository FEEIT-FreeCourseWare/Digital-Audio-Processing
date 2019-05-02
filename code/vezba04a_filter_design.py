#!/usr/bin/env python3
#
# Copyright 2017 - 2019 by Branislav Gerazov
#
# See the file LICENSE for the license associated with this software.
#
# Author(s):
#   Branislav Gerazov, March 2017

"""
Digital Audio Systems

Excercise 04: Filter Design using combination of DFT and window methods.

@author: Branislav Gerazov
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy import fftpack as fftp
from scipy import signal as sig
import os
import das

# %% load audio
folder = 'audio/'
filename = 'Mara.wav'
fs, wav = wavfile.read(folder+filename)
wav = wav / 2**15
t = np.arange(wav.shape[0]) / fs
os.system('play '+folder+filename)

# %% define ideal LP filter
fs = 44100
f_l = 5000 / (fs/2)  # normalized cut-off frequency
n_fft = fs

w = np.linspace(0, np.pi, n_fft//2+1)

H_lp = np.zeros(w.size)
H_lp[w < f_l * np.pi] = 1

H_lp_2p = np.r_[H_lp, H_lp[-2:0:-1]]
w_2p = np.linspace(0, 2*np.pi, n_fft, endpoint=False)

# %% plot
plt.figure()
plt.plot(w, H_lp, lw=4, alpha=.5)
plt.plot(w_2p, H_lp_2p, lw=2, alpha=.5)

# %% calculate impulse response using DFT
h_lp = fftp.ifft(H_lp, n_fft)
h_lp = fftp.fftshift(h_lp)

# %% window it
n = 128 + 1
n_h = int((n-1)/2)
n_ffth = int(n_fft/2)
win = sig.get_window('boxcar', n)
n_win = np.arange(n_ffth-n_h, n_ffth+n_h+1, dtype='int')
h_rect = h_lp[n_win] * win

# %% plot designed filters
plt.figure()
plt.subplot(211)
plt.plot(h_lp, lw=2, alpha=.5)
plt.plot(n_win, h_rect, lw=1, alpha=.9)
plt.grid()
plt.axis([n_ffth-1.5*n_h, n_ffth+1.5*n_h, -0.1, 0.25])

plt.subplot(212)
f, h_lp_spec = das.get_spectrum(fs, h_lp, n_fft)
f, h_rect_spec = das.get_spectrum(fs, h_rect, n_fft)
plt.plot(f, h_lp_spec - np.max(h_lp_spec))
plt.plot(f, h_rect_spec - np.max(h_rect_spec))
plt.grid()
plt.axis([0, 10000, -90, 10])
