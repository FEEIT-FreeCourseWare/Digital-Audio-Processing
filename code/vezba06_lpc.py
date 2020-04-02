#!/usr/bin/env python3
#
# Copyright by Branislav Gerazov 2016 - 2020
#
# See the file LICENSE for the license associated with this software.
#
# Author(s):
#   Branislav Gerazov, Apr 2016 - 2020

"""
Digital Audio Processing

Excercise 06: Linear Predictive Coding.

@author: Branislav Gerazov
"""
import numpy as np
from matplotlib import pyplot as plt
from math import pi
from scipy.io import wavfile
from scipy import signal as sig
from scikits.talkbox import lpc
import dap

#%% load wav
fs, wav = wavfile.read('audio/glas_aaa.wav')
wav = wav / 2**15
# wav = wav * sig.hamming(wav.size)
f, wav_spec = dap.get_spectrum(wav, fs)

#%% get LPC parameters
a_lp, e, k = lpc(wav, 25)
b_inv = np.concatenate(([0],-a_lp[1:]))
wav_est = sig.lfilter(b_inv,1, wav)
wav_err = wav - wav_est
G = e
f, err_spec = dap.get_spectrum(wav_err, fs)

#%% plot
#plt.figure()
#plt.plot(wav)
#plt.plot(est_wav)
#plt.figure()
#plt.plot(err)

#%% LP filter impulse response and transfer function
x = np.zeros(.02*fs)
x[0] = 1
h_lp = sig.lfilter(G, a_lp, x)
w, H_lp = sig.freqz(G, a_lp)
f_lp = w / pi * fs/2
H_lp = 20*np.log10(np.abs(H_lp))

#%% plot
t = np.arange(0, wav.size/fs, 1/fs)
t_lp = np.arange(0, h_lp.size/fs, 1/fs)
plt.figure()
plt.subplot(311)
plt.plot(t, wav, alpha=0.8)
plt.plot(t, wav_est, alpha=0.8)
plt.plot(t, wav_err*10, alpha=0.8)
plt.grid()
plt.axis([0.2, .212, -.7, .6])
plt.subplot(312)
plt.plot(t_lp, h_lp, 'g')
plt.grid()
plt.axis([-.0001, .012, -.0005, .0005])
plt.subplot(313)
plt.plot(f, wav_spec, 'b', alpha=0.8)
plt.plot(f, err_spec, 'r', linewidth=1, alpha=0.8)
plt.plot(f_lp, H_lp, 'g', linewidth=3, alpha=0.8)
plt.grid()
plt.axis([0, 15000, -100, 0])

#%% synthesis
