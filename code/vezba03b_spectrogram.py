#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Copyright 2017 by Branislav Gerazov
#
# See the file LICENSE for the license associated with this software.
#
# Author(s):
#   Branislav Gerazov, March 2017

"""
Digital Audio Systems

Excercise 03: Spectrogram.

@author: Branislav Gerazov
"""

from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
import os
from scipy import signal as sig 
import das

#%% load audio
fs, wav = wavfile.read('audio/zvona2.wav')
#os.system('play audio/zvona2.wav')
wav = wav / 2**15
t = np.arange(0, wav.size/fs, 1/fs)
f, wav_amp = das.get_spectrum(fs, wav, plot=True)

#%% aliasing
# wavfile.write('audio/Zvona_alias.wav', fs/4, 
#               np.array(wav[0:-1:4] * 2**15, dtype='int16'))

#%% extract spectrogram
# define window
#Nt = .050  # ms
#N = int(Nt*fs)
N = 2048  # 2**11
Nh = int(N/2)
H = Nh  # hop size
win = sig.get_window('hamming',N)

# windowing
poz = Nh  # pozicija na sredinata na prozorecot
pad = np.zeros(Nh)
wav_pad = np.r_[pad, wav, pad]
while poz < wav_pad.size-Nh:
    frame = wav_pad[poz-Nh : poz+Nh] * win
    f_frame, frame_spec = das.get_spectrum(frame, fs)
    if poz == Nh:
        frames =frame[:,np.newaxis]
        frames_spec = np.array([frame_spec]).T
    else:
        frames = np.hstack((frames, frame[:,np.newaxis]))
        frames_spec = np.hstack((frames_spec,\
                      np.array([frame_spec]).T))
    poz += H                      

no_frames = frame_specs.shape[1]
t_frames = np.arange(no_frames) * H / fs

#%% plot frames 
plt.figure()
plt.imshow(np.abs(frames), 
           extent=[0, t_frames[-1], 0, f_frame[-1]],
           aspect='auto',
           origin='lower',
           vmin=0,
           vmax=.25,
           cmap='viridis')
plt.colorbar()

#%% plot spectrogram
plt.imshow(frames_spec, aspect='auto', 
           origin='lower', 
           extent=[0, t[-1], 0, f_frame[-1]],
           vmin=-100,vmax=0, 
           cmap='viridis')
cbar = plt.colorbar()
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
cbar.ax.set_ylabel('Amplitude [dB]')
plt.axis([0, t[-1], 0, 10000])

#%% 
das.get_spectrogram(fs, wav, 256, win='hann')
das.get_spectrogram(fs, wav, 2048, win='hann')
das.get_spectrogram(fs, wav, 16384, win='hann')

#%% 
das.get_spectrogram(fs, wav, 2048, win='boxcar')
das.get_spectrogram(fs, wav, 2048, win='hann')
das.get_spectrogram(fs, wav, 2048, win='hamming')
das.get_spectrogram(fs, wav, 2048, win='blackmanharris')
