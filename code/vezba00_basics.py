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

Excercise 00: Basics of working with sound.

@author: Branislav Gerazov
"""
from __future__ import division
import numpy as np
from matplotlib import pyplot as plt 
from scipy.io import wavfile
import os

#%% load
fs, wav = wavfile.read('audio/Skopsko_stereo.wav')
wav = wav / 2**15
t = np.arange(wav.shape[0]) / fs

#%% plot
plt.figure()
plt.subplot(2,1,1)
plt.plot(t, wav[:,0])
plt.subplot(2,1,2)
plt.plot(t, wav[:,1])

#%% play
os.system('play audio/Skopsko_stereo.wav')

#%% mono
wav_mono = np.mean(wav,1)

#%% short
wav_short = wav_mono[0:4*fs]
t = np.arange(wav_short.shape[0]) / fs

#%% plot
plt.figure()
plt.plot(t, wav_short)

#%% save and play
wavfile.write('audio/skopsko_short.wav', fs, 
              np.array(wav_short*2**15,dtype='int16'))
os.system('play audio/skopsko_short.wav')

#%% amplitude
wav_glasno = wav_short * 4
wav_tivko = wav_short * .25

#%% plot
plt.figure()
plt.plot(t, wav_glasno)
plt.plot(t, wav_short)
plt.plot(t, wav_tivko)
plt.grid()
plt.legend(['glasno','orig','tivko'])

#%% save and play
wavfile.write('audio/skopsko_tivko.wav', fs, 
              np.array(wav_tivko*2**15,dtype='int16'))
os.system('play audio/skopsko_tivko.wav')

#%% save and play
wavfile.write('audio/skopsko_glasno.wav', fs, 
              np.array(wav_glasno*2**15,dtype='int16'))
os.system('play audio/skopsko_glasno.wav')

#%% distortion
wav_distortion = np.copy(wav_glasno)
wav_distortion[wav_distortion > 1] = 1
wav_distortion[wav_distortion < -1] = -1

#%% plot
plt.figure()
plt.plot(t, wav_glasno)
plt.plot(t, wav_distortion)
plt.grid()

#%% normalisation
# to +/-1
wav_norm = wav_glasno / np.max(np.abs(wav_glasno))
wav_norm2 = wav_short / np.max(np.abs(wav_short))
np.all(wav_norm == wav_norm2)

#%% normalization func
# level_dB = 20*np.log10(level)
def normalise(wav, level_dB=0):
    level = 10**(level_dB/20)
    wav_norm = wav / np.max(np.abs(wav)) * level
    return wav_norm
    
#%%
wav_norm0 = normalise(wav_short)
wav_norm3 = normalise(wav_short, -3)
wav_norm18 = normalise(wav_short, -18)

#%% move functuion to dedicated utility module
import das
wav_norm = das.normalise(wav_short,-60)


