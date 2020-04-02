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

Excercise 03: Spectrogram.

@author: Branislav Gerazov
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy import signal as sig
import os
import dap

# %% load audio
fs, wav = wavfile.read('audio/zvona2.wav')
os.system('play audio/zvona2.wav')
wav = wav / 2**15
t = np.arange(0, wav.size/fs, 1/fs)
f, wav_amp = dap.get_spectrum(fs, wav, plot=True)

# %% aliasing
# wavfile.write('audio/Zvona_alias.wav', fs/4,
#               np.array(wav[0:-1:4] * 2**15, dtype='int16'))

# %% extract spectrogram
# define window
# t_win = .050  # ms
# n_win = int(t_win * fs)
n_win = 2048
win = sig.get_window('hamming', n_win)
n_half = n_win // 2
n_hop = n_half
# pad signal
pad = np.zeros(n_half)
wav_pad = np.concatenate((pad, wav, pad))
# loop
pos = 0
spectrogram = None
while pos <= wav_pad.size - n_win:
    frame = wav_pad[pos: pos+n_win]
    f_frame, frame_spec = dap.get_spectrum(fs, wav, n_fft=n_win)
    frame_spec = frame_spec[:, np.newaxis]
    if spectrogram is None:
        spectrogram = frame_spec
    else:
        spectrogram = np.concatenate((spectrogram, frame_spec), axis=1)
    pos += n_hop
n_frame = spectrogram.shape[1]
t_frame = np.arange(n_frame) * n_hop/fs

# %% plot frames
plt.figure()
plt.imshow(spectrogram, aspect='auto', origin='lower',
           extent=[0, t_frame[-1], 0, f_frame[-1]],
           vmin=-100, vmax=0,
           cmap='viridis')
cbar = plt.colorbar()
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
cbar.ax.set_ylabel('Amplitude [dB]')
# plt.axis([0, t[-1], 0, 10000])

# %% spectrograms for different n_win
dap.get_spectrogram(fs, wav, 256, win_type='hann')
dap.get_spectrogram(fs, wav, 2048, win_type='hann')
dap.get_spectrogram(fs, wav, 16384, win_type='hann')

# %% spectrograms for different win_type
dap.get_spectrogram(fs, wav, 2048, win_type='boxcar')
dap.get_spectrogram(fs, wav, 2048, win_type='hann')
dap.get_spectrogram(fs, wav, 2048, win_type='hamming')
dap.get_spectrogram(fs, wav, 2048, win_type='blackmanharris')
