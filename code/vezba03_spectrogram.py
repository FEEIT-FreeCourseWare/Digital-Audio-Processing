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

#%% load wav
fs, wav = wavfile.read('audio/zvona2.wav')
#os.system('play audio/wav.wav')
wav = wav / 2**15
t = np.arange(0, wav.size/fs, 1/fs)
f, wav_amp = das.get_spectrum(wav, fs)

#%% aliasing
#fs = fs/4
# wav = wav[0:-1:4]
# wavfile.write('audio/Zvona_alias.wav', fs, wav)

#%% extract spectrogram
# define window
M = wav.size
#Nt = .050  # ms
#N = np.round(Nt * fs)
N = 2048  # 2**11
Nh = int(N/2)
H = int(N/2)  # hop size
win = sig.get_window('hamming',N)

# windowing
poz = Nh  # pozicija na sredinata na prozorecot
pad = np.zeros(Nh)
wav_pad = np.concatenate((pad, wav, pad))
M = M + N  # wav_pad length
while poz < M-Nh:
    frame = wav_pad[poz-Nh : poz+Nh] * win
    f_frame, frame_spec = das.get_spectrum(frame, fs)
    if poz == Nh:
        frames_spec = np.array([frame_spec]).T
    else:
        frames_spec = np.hstack((frames_spec,\
                      np.array([frame_spec]).T))
    poz += H                      

#%% plot
plt.figure(figsize=(3,2
                    ))
plt.imshow(frames_spec, aspect='auto', \
           origin='lower', \
           extent=[0, t[-1], 0, f_frame[-1]],\
           vmin=-100,vmax=0, cmap='viridis')
cbar = plt.colorbar()
#plt.xlabel('Time [s]')
#plt.ylabel('Frequency [Hz]')
#cbar.ax.set_ylabel('Amplitude [dB]')
plt.axis([0, t[-1], 0, 10000])
