#!/usr/bin/env python3
#
# Copyright 2017 - 2019 by Branislav Gerazov
#
# See the file LICENSE for the license associated with this software.
#
# Author(s):
#   Branislav Gerazov, Mar 2017

"""
Digital Audio Systems

Excercise 05: Digital Audio Effects: Robotiser.

@author: Branislav Gerazov
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal as sig
from scipy import fftpack as fft
import os
import das

# %% load wave
path = 'audio/'
file_name = 'Solzi.wav'
os.system('play '+path+file_name)
fs, wav = wavfile.read(path+file_name)
wav = wav / 2**15
t = np.arange(wav.size) / fs

# %% filter design
order = 5
bandwidth = 500
amp = 1000
offset = 1500
f = 1
n_win = int(.040 * fs)
n_half = n_win // 2
n_hop = n_half
n_pad = n_half
pad = np.zeros(n_pad)
wav_pad = np.concatenate((pad, wav, pad))
t_frame = np.arange(0, wav_pad.size, n_hop) / fs
f_cs = -amp * np.sin(2*np.pi*f*t_frame) + offset
plt.plot(t_frame, f_cs)

# %% generate filters
b_frame = []
a_frame = []
for f_c in f_cs:
    f_l = f_c - bandwidth/2  # / (fs/2)
    f_h = f_c + bandwidth/2  # / (fs/2)
    b, a = sig.iirfilter(order, [f_l, f_h],
                         btype='bandpass', ftype='butter', fs=fs)
    b_frame.append(b)
    a_frame.append(a)

# %% window + wah
win = sig.get_window('hann', n_win)
t_win = n_win / fs
pos = 0
i = 0
wav_wah = np.zeros(wav_pad.size)
while pos <= wav_pad.size - n_win:
    frame = wav_pad[pos : pos+n_win]
    frame = frame * win
    frame_wah = sig.lfilter(b_frame[i], a_frame[i], frame)
    wav_wah[pos : pos+n_win] += frame_wah * win
    pos += n_hop
    i += 1

# %% wav write and play
wav_wah = 0.1*wav_pad + 0.9*wav_wah
wav_wah = das.normalise(wav_wah, -3)
wav_wah_int16 = wav_wah * 2**15
wav_wah_int16 = wav_wah_int16.astype('int16')
wavfile.write(path+file_name+'_wah.wav', fs, wav_wah_int16)
os.system('play '+path+file_name+'_wah.wav')

# %% plot spectrogram
__ = das.get_spectrogram(fs, wav_pad, 2048, win_type='hann', plot=True)
__ = das.get_spectrogram(fs, wav_wah, 2048, win_type='hann', plot=True)
