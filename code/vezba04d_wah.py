#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Copyright 2017 by Branislav Gerazov
#
# See the file LICENSE for the license associated with this software.
#
# Author(s):
#   Branislav Gerazov, Mar 2017

"""
Digital Audio Systems

Excercise 04: Wah-wah.

@author: Branislav Gerazov
"""
from __future__ import division
import numpy as np
from matplotlib import pyplot as plt 
from scipy.io import wavfile
import os
from scipy import fftpack as fftp
from scipy import signal as sig
import das

#%% load wav
folder = 'audio/'
filename = 'Solzi.wav'
fs, wav = wavfile.read(folder+filename)
wav = wav / 2**15
t = np.arange(wav.shape[0]) / fs
os.system('play '+folder+filename)

#%% define filter bandwidth[n]
f_wah = 2
B_wah = 1000
f_c_l = 1000
f_c_h = 4000

f_cs = (np.sin(2*np.pi*f_wah*t)+1)/2 * (f_c_h - f_c_l) + \
      f_c_l  
f_ls = f_cs - B_wah/2
f_hs = f_cs + B_wah/2

plt.figure()
plt.plot(t, f_cs)
plt.plot(t, f_ls)
plt.plot(t, f_hs)

#%% get frames from signal
win_t = 50  # ms
win_s = int(win_t * .001 * fs) // 2 * 2
t_frames, f_frame, frame_specs, frames = \
             das.get_spectrogram(fs, wav, 
                                 N=win_s, 
                                 win='hann', 
                                 plot=False)

#%% generate filter coefficients[n]
order = 5
bbb = []
aaa = []
for t_frame in t_frames:
    f_l = f_ls[t == t_frame]
    f_h = f_hs[t == t_frame]
    b, a = sig.iirfilter(order, 
                         [f_l/(fs/2), f_h/(fs/2)],
                         btype='bandpass',
                         ftype='butter')
    bbb.append(b)
    aaa.append(a)
    
#%% filter frames 
frames_filt = np.empty_like(frames)
#for frame, b, a in zip(frames.T, bbb, aaa):
for i in range(frames.shape[1]):
    frame = frames[:,i]
    b = bbb[i]
    a = aaa[i]
    frame_filt = sig.lfilter(b, a, frame)
    frames_filt[:,i] = frame_filt

#%% Overlap add 
no_frames = frames.shape[1]
H = int(win_s / 2)
wav_filt = np.zeros((no_frames+1)*H)
pos = 0
for frame_filt in frames_filt.T:
    wav_filt[pos:pos+win_s] = wav_filt[pos:pos+win_s] + \
                              frame_filt
    pos += H
  
#%% compare spectrograms
das.get_spectrogram(fs, wav)
das.get_spectrogram(fs, wav_filt)

#%% mix
wav_pad = np.r_[np.zeros(H), wav, 
            np.zeros(wav_filt.size - wav.size - H)]
wav_out = 0.3*wav_pad + 0.7*wav_filt
das.get_spectrogram(fs, wav_out)

#%% play
wavfile.write('audio/Solzi_wah.wav', fs,
              np.array(wav_out*2**15, 
                       dtype='int16'))
os.system('play audio/Solzi_wah.wav')



