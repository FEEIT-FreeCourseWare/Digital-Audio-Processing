#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Copyright 2016 by Branislav Gerazov
#
# See the file LICENSE for the license associated with this software.
#
# Author(s):
#   Branislav Gerazov, Apr 2016

"""
Digital Audio Systems

Excercise 04: Filters and equalization.

@author: Branislav Gerazov
"""
from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from math import pi
from scipy.io import wavfile
import das
from scipy import fftpack as fp
from scipy import signal as sig

#%% load wav
fs, wav = wavfile.read('audio/Mara.wav')

#%% generate filters
# bass
f_b = 400 / (fs/2)
#p = 101
p = 7
#b = sig.firwin(p, f_l, window='hamming',nyq=1)
b_b, a_b = sig.iirfilter(p, f_b, btype='low', ftype='butter')
w, H_b = sig.freqz(b_b,a_b)
f = w /pi * (fs/2)
H_b = 20*np.log10(H_b)

# mid
f_ml = 400 / (fs/2)
f_mh = 4000 / (fs/2)
p = 7
b_m, a_m = sig.iirfilter(p, [f_ml, f_mh], btype='band', ftype='butter')
w, H_m = sig.freqz(b_m,a_m)
f = w /pi * (fs/2)
H_m = 20*np.log10(H_m)

# high
f_h = 4000 / (fs/2)
p = 7
b_t, a_t = sig.iirfilter(p, f_h, btype='high', ftype='butter')
w, H_t = sig.freqz(b_t,a_t)
f = w /pi * (fs/2)
H_t = 20*np.log10(H_t)
#%% plot transfer functions

plt.figure()
plt.subplot(212)
plt.plot(f, H_b)
plt.plot(f, H_m)
plt.plot(f, H_t)
plt.axis([0, 10000, -60, 10])
plt.grid()

plt.subplot(211)
x = np.zeros(1000)
x[0] = 1
h_b = sig.lfilter(b_b, a_b, x)
h_m = sig.lfilter(b_m, a_m, x)
h_t = sig.lfilter(b_t, a_t, x)
plt.plot(h_b)
plt.plot(h_m)
plt.plot(h_t)
plt.axis([0, 200, -.2, .34])
plt.grid()

#%% apply filters
#wav_bass = sig.filtfilt(b_b,a_b, wav)
wav_bass = sig.lfilter(b_b,a_b, wav)
wav_mid = sig.lfilter(b_m,a_m, wav)
wav_treble = sig.lfilter(b_t,a_t, wav)

#%% apply equalization
g_bass = -20  # dB
g_mid = 0  # dB
g_treble = 20  # dB
g_b = 10**(g_bass/20)
g_m = 10**(g_mid/20)
g_t = 10**(g_treble/20)
wav_out = g_b*wav_bass + g_m*wav_mid + g_t*wav_treble
#wavfile.write('skopsko_bass.wav', fs, np.array(wav_bass, dtype='int16'))

#%% plot spectrum
f, wav_spec = das.get_spectrum(wav, fs)
f, wav_bass_spec = das.get_spectrum(wav_bass, fs)
f, wav_mid_spec = das.get_spectrum(wav_mid, fs)
f, wav_treble_spec = das.get_spectrum(wav_treble, fs)

plt.figure()
plt.plot(f, wav_bass_spec-np.max(wav_bass_spec))
plt.plot(f, wav_mid_spec-np.max(wav_bass_spec))
plt.plot(f, wav_treble_spec-np.max(wav_bass_spec))
plt.axis([10,10000,-70, 10])
plt.grid()

#%% plot wav and wav_out
plt.figure()
f, wav_out_spec = das.get_spectrum(wav_out, fs)
plt.plot(f, wav_spec, 'b', alpha=.5)
plt.plot(f, wav_out_spec, 'r', alpha=.5)
plt.grid()

plt.show()
#%% play
#import os
#wav_out = wav_out / np.max(np.abs(wav_out))
#wavfile.write('audio/Mara.wav', fs, 
#              np.array(wav_out*2**15, dtype='int16'))
#os.system('play audio/Mara.wav')
#os.system('play audio/Mara_eql.wav')