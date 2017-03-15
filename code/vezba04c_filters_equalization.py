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

Excercise 04: Filter bank equalization.

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
filename = 'Mara.wav'
fs, wav = wavfile.read(folder+filename)
wav = wav / 2**15
t = np.arange(wav.shape[0]) / fs
#os.system('play '+folder+filename)

#%% iir filterbank
order = 7
f_l = 400
f_h = 4000
b_lp, a_lp = sig.iirfilter(order, 
                     f_l/(fs/2),
                     btype='lowpass',
                     ftype='butter')

b_bp, a_bp = sig.iirfilter(order, 
                     [f_l/(fs/2), f_h/(fs/2)],
                     btype='bandpass',
                     ftype='butter')

b_hp, a_hp = sig.iirfilter(order, 
                     f_h/(fs/2),
                     btype='highpass',
                     ftype='butter')

#%% plot freqz
w, H_lp = sig.freqz(b_lp, a_lp)
w, H_bp = sig.freqz(b_bp, a_bp)
w, H_hp = sig.freqz(b_hp, a_hp)

plt.figure()
plt.plot(w/np.pi*fs/2, 
         20*np.log10(np.abs(H_lp)))
plt.plot(w/np.pi*fs/2, 
         20*np.log10(np.abs(H_bp)))
plt.plot(w/np.pi*fs/2, 
         20*np.log10(np.abs(H_hp)))
plt.grid('on')

#%% filter signal
wav_lp = sig.lfilter(b_lp, a_lp, wav)
wav_bp = sig.lfilter(b_bp, a_bp, wav)
wav_hp = sig.lfilter(b_hp, a_hp, wav)

#%% gains
G_lp = 24  # dB
G_bp = -24  # dB
G_hp = 24  # dB

g_lp = 10**(G_lp/20)
g_bp = 10**(G_bp/20)
g_hp = 10**(G_hp/20)

#%% mix
wav_out = wav_lp * g_lp + \
          wav_bp * g_bp + \
          wav_hp * g_hp

wav_out = das.normalise(wav_out)

#%% play
wavfile.write('audio/Mara_eq.wav', fs,
              np.array(wav_out*2**15, 
                       dtype='int16'))
os.system('play audio/Mara_eq.wav')

#%% compare to original
os.system('play audio/Mara.wav')

#%% compare spectrograms
das.get_spectrogram(fs, wav)
das.get_spectrogram(fs, wav_out)
