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

Excercise 02: Spectrum.

@author: Branislav Gerazov
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

#%% load WAV file

filename = 'audio/Viluska_440Hz.wav'
fs, viluska = wavfile.read(filename)
#os.system('play ' + filename)

#%% convert to float, generate t vector 
viluska = viluska / 2**15
N = viluska.size
t = np.arange(0, N)
t = t / fs

#%% plot
plt.figure(figsize=(15,5))
plt.subplot(121)  # 1x2 графици, прв график
plt.plot(t, viluska)
plt.grid()
plt.axis([0, t[-1], -1, 1])  # [xmin, xmax, ymin, ymax]
plt.subplot(122)  # 1x2 графици, втор график
plt.plot(t, viluska)
plt.grid()
plt.axis([0.5, 0.55, -1, 1])  # [xmin, xmax, ymin, ymax]

#%% Spectral analysis
from scipy import fftpack as ffp

Nfft = 2**np.ceil(np.log(N)/np.log(2))
Nh = Nfft / 2

viluska_fft = ffp.fft(viluska, Nfft)
viluska_fft = viluska_fft[0:Nh+1]
viluska_amp = np.abs(viluska_fft) /N
viluska_amp[1:-1] = viluska_amp[1:-1] * 2
viluska_log = 20*np.log10(viluska_amp)
viluska_ph = np.angle(viluska_fft)
viluska_ph = np.unwrap(viluska_ph)

#w = np.arange(0, 2*pi, 2*pi/N)
f = np.arange(0, Nh+1)
f = f / Nfft *fs

#%% plot
plt.figure()
plt.subplot(311)
plt.plot(t, viluska)
plt.axis([0, t[-1], -1, 1])  # [xmin, xmax, ymin, ymax]
plt.grid()
plt.subplot(312)
plt.plot(f, viluska_amp)
plt.axis([0, f[-1], np.min(viluska_amp), np.max(viluska_amp)])
#plt.plot(w_shift, np.fft.fftshift(viluska_amp))
plt.grid()
plt.subplot(313)
plt.plot(f, viluska_ph)
plt.axis([0, f[-1], np.min(viluska_ph), np.max(viluska_ph)])
##plt.plot(w_shift, np.fft.fftshift(viluska_ph))
plt.grid()