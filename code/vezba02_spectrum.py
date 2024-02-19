#!/usr/bin/env python3
#
# Copyright by Branislav Gerazov 2017 - 2020
#
# See the file LICENSE for the license associated with this software.
#
# Author(s):
#   Branislav Gerazov, March 2017 - 2019

"""
Digital Audio Processing

Excercise 02: Spectrum.

@author: Branislav Gerazov
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import fftpack as fft
import os

# %% load audio
audio_path = 'audio/'
file_name = 'viluska.wav'  # tone A
fs, wav = wavfile.read(audio_path + file_name)
os.system('play ' + audio_path + file_name)

# %% convert to -1 to 1 float, generate t vector
wav = wav / 2**15
n = wav.size
ts = 1 / fs
t = np.arange(0, n) * ts

# %% plot
plt.figure(figsize=(15, 5))
plt.plot(t, wav)
plt.grid()

# %% get spectrum
x = np.ceil(np.log2(n))
n_fft = int(2**x)
wav_fft = fft.fft(wav, n_fft)

wav_amp = np.abs(wav_fft)
wav_amp = wav_amp / n
n_keep = int(n_fft/2) + 1
wav_amp = wav_amp[: n_keep]
wav_amp[1: -1] = 2 * wav_amp[1: -1]

wav_pha = np.angle(wav_fft)
wav_pha = np.unwrap(wav_pha)
wav_pha = wav_pha[: n_keep]

# %% FFT bin frequencies
# whole spectrum
# w = np.linspace(0, 2*np.pi, n_fft, endpoint=False)
# f = np.linspace(0, fs, n_fft, endpoint=False)
# kept spectrum
# w = np.linspace(0, np.pi, n_keep)
f = np.linspace(0, fs/2, n_keep)

# %% plot
plt.figure()
plt.subplot(211)
plt.plot(f, wav_amp)
plt.grid()
plt.subplot(212)
plt.plot(f, wav_pha)
plt.grid()
