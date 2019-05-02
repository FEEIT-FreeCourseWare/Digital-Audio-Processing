#!/usr/bin/env python3
#
# Copyright 2016 - 2019 by Branislav Gerazov
#
# See the file LICENSE for the license associated with this software.
#
# Author(s):
#   Branislav Gerazov, Apr 2016

"""
Digital Audio Systems

Excercise 05: Digital Audio Effects: Echo.

@author: Branislav Gerazov
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy import signal as sig
import os

# %% define parameters
# FIR echo
fs = 22050
Dt = 0.5  # sec
D = int(Dt*fs)  # samples
b_D = 0.5
b_fir = np.zeros(D+1)
b_fir[0] = 1
b_fir[D] = b_D

# multiple FIR echo
D0t = 0.2
D0 = int(D0t*fs)  # samples
D1t = 0.3
D1 = int(D1t*fs)  # samples
b_fir_mul = b_fir.copy()
b_fir_mul[D0] = 0.4
b_fir_mul[D1] = 0.3

# infinite IIR echo
a_iir = np.zeros(D+1)
a_iir[0] = 1
a_iir[D] = b_D

# allpass echo
b_ap = np.zeros(D+1)
b_ap[0] = b_D
b_ap[D] = 1
a_ap = a_iir.copy()

# FIR
w, H_fir = sig.freqz(b_fir, [1])
f = w / np.pi * fs/2
H_fir = 20*np.log10(np.abs(H_fir))

# FIR_mul
w, H_fir_mul = sig.freqz(b_fir_mul, [1])
H_fir_mul = 20*np.log10(np.abs(H_fir_mul))

# IIR
x = np.zeros(5*fs)
x[0] = 1
h_iir = sig.lfilter([1], a_iir, x)
w, H_iir = sig.freqz([1], a_iir)
H_iir = 20*np.log10(np.abs(H_iir))

# AP
h_ap = sig.lfilter(b_ap, a_ap, x)
w, H_ap = sig.freqz(b_ap, a_ap)
H_ap = 20*np.log10(np.abs(H_ap))

# %% plot
t = np.arange(0, b_fir.size/fs, 1/fs)
plt.figure()
plt.subplot(421)
t = np.arange(0, b_fir.size/fs, 1/fs)
plt.plot(t, b_fir)
plt.grid()
plt.axis([-.02, t[-1]+.02, -.1, 1.1])
plt.subplot(422)
plt.plot(f, H_fir)
plt.grid()
plt.axis([0, 1000, -7, 7])

plt.subplot(423)
plt.plot(t, b_fir_mul)
plt.axis([-.02, t[-1]+.02, -.1, 1.1])
plt.grid()
plt.subplot(424)
plt.plot(f, H_fir_mul)
plt.axis([0, 1000, -7, 7])
plt.grid()

t = np.arange(0, h_iir.size/fs, 1/fs)
plt.subplot(425)
plt.plot(t, h_iir)
plt.axis([-.2, t[-1]+.2, -.6, 1.1])
plt.grid()
plt.subplot(426)
plt.plot(f, H_iir)
plt.axis([0, 1000, -7, 7])
plt.grid()

plt.subplot(427)
plt.plot(t, h_ap)
plt.axis([-.2, t[-1]+.2, -.4, .8])
plt.grid()
plt.subplot(428)
plt.plot(f, H_ap)
plt.axis([0, 1000, -7, 7])
plt.grid()

# %% apply echo
fs, wav = wavfile.read('audio/Pato_22K.wav')
wav_eho = sig.lfilter(b_fir, [1], wav)
wav_eho = sig.lfilter(b_fir_mul, [1], wav)
wav_eho_iir = sig.lfilter([1], a_iir, wav)
wav_eho_ap = sig.lfilter(b_ap, a_ap, wav)

# %% play
os.system('play audio/Pato_22K.wav')
wavfile.write('audio/Pato_eho_fir.wav', fs, np.array(wav_eho, dtype='int16'))
os.system('play audio/Pato_eho_fir.wav')
wavfile.write('audio/Pato_eho_fir_mul.wav', fs, np.array(wav_eho,
                                                         dtype='int16'))
os.system('play audio/Pato_eho_fir_mul.wav')
wavfile.write('audio/Pato_eho_iir.wav', fs, np.array(wav_eho, dtype='int16'))
os.system('play audio/Pato_eho_iir.wav')
wavfile.write('audio/Pato_eho_ap.wav', fs, np.array(wav_eho, dtype='int16'))
os.system('play audio/Pato_eho_ap.wav')
