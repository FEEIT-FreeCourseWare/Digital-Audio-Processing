#!/usr/bin/env python3
#
# Copyright by Branislav Gerazov 2017 - 2020
#
# See the file LICENSE for the license associated with this software.
#
# Author(s):
#   Branislav Gerazov, Mar 2017 - 2020

"""
Digital Audio Processing

Excercise 04: Notch filter.

@author: Branislav Gerazov
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
import os
from scipy import signal as sig
import dap

# %% load wav
folder = 'audio/'
file_name = 'Pato_8k.wav'
fs, wav = wavfile.read(folder+file_name)
wav = wav / 2**15
t = np.arange(wav.shape[0]) / fs
os.system('play '+folder+file_name)

# %% brum
brum = np.sin(2*np.pi*50*t) + \
       .05 * np.sin(2*np.pi*100*t) + \
       .01 * np.sin(2*np.pi*150*t)

# %% add brum
wav_brum = wav + .02*brum
wav_brum = dap.normalise(wav_brum)

# %% plot
plt.figure()
plt.plot(t, wav)
plt.plot(t, wav_brum)

# %% play
wavfile.write('audio/Pato_brum.wav', fs,
              np.array(wav_brum*2**15, dtype='int16'))
os.system('play audio/Pato_brum.wav')

# %% design notch filter
r = 0.99
f0 = 50
w0 = f0 / (fs/2) * np.pi
b_notch = [1, -2*np.cos(w0), 1]
a_notch = [1, -2*r*np.cos(w0), r**2]

w, H_notch = sig.freqz(b_notch, a_notch)
plt.figure()
plt.plot(w/np.pi*fs/2,
         20*np.log10(np.abs(H_notch)))
plt.grid('on')

# %% filetr
wav_notch = sig.lfilter(b_notch, a_notch, wav_brum)

# %% compare spectrograms
dap.get_spectrogram(fs, wav, 256)
dap.get_spectrogram(fs, wav_notch, 256)

# %% compare spectrums
f, wav_spec = dap.get_spectrum(fs, wav_brum)
f, wav_notch_spec = dap.get_spectrum(fs, wav_notch)

plt.figure()
plt.plot(f, wav_spec)
plt.plot(f, wav_notch_spec)
plt.grid()

# %% play
wavfile.write('audio/Pato_notch.wav', fs,
              np.array(wav_notch*2**15,
                       dtype='int16'))
os.system('play audio/Pato_notch.wav')
