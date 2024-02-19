#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright by Branislav Gerazov 2017 - 2020
#
# See the file LICENSE for the license associated with this software.
#
# Author(s):
#   Branislav Gerazov, March 2017 - 2020

"""
Digital Audio Processing

Excercise 00: Basics of working with sound.

@author: Branislav Gerazov
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
import os

# %% load
audio_path = 'audio/'
wav_name = 'Skopsko_stereo.wav'
fs, wav = wavfile.read(audio_path + wav_name)
wav = wav / 2**15
t = np.arange(wav.shape[0]) / fs

# %% plot
plt.figure()
plt.subplot(211)
plt.plot(t, wav[:, 0])
plt.grid()
plt.subplot(212)
plt.plot(t, wav[:, 1])
plt.grid()

# %% play
os.system('play ' + audio_path + wav_name)

# %% mono
wav_mono = np.mean(wav, axis=1)

# %% short
wav_short = wav_mono[0: 4*fs]
t = np.arange(wav_short.shape[0]) / fs

# %% plot
plt.figure()
plt.plot(t, wav_short)
plt.grid()

# %% save and play
wavfile.write(
    audio_path + 'skopsko_short.wav', fs, np.int16(wav_short * 2**15)
    )
os.system(f'play {audio_path}skopsko_short.wav')

# %% amplitude
wav_loud = wav_short * 4
wav_quiet = wav_short * 0.25

# %% plot
plt.figure()
plt.plot(t, wav_loud)
plt.plot(t, wav_short)
plt.plot(t, wav_quiet)
plt.grid()
plt.legend(['loud', 'orig', 'quiet'])

# %% save and play
wavfile.write(
    audio_path + 'skopsko_quiet.wav', fs, np.int16(wav_quiet * 2**15)
    )
os.system(f'play {audio_path}skopsko_quiet.wav')

# %% save and play
wavfile.write(
    audio_path + 'skopsko_loud.wav', fs, np.int16(wav_loud * 2**15)
    )
os.system(f'play {audio_path}skopsko_loud.wav')

# %% distortion
wav_distortion = wav_loud.copy()
wav_distortion[wav_distortion > 1] = 1
wav_distortion[wav_distortion < -1] = -1

# %% plot
plt.figure()
plt.plot(t, wav_loud)
plt.plot(t, wav_distortion)
plt.grid()

# %% normalisation
# to +/-1
wav_norm = wav_loud / np.max(np.abs(wav_loud))
wav_norm2 = wav_short / np.max(np.abs(wav_short))
assert np.all(wav_norm == wav_norm2)


# %% normalization func
def normalise(wav, level_dB=0):
    # level_dB = 20*np.log10(level)
    level = 10**(level_dB/20)
    wav_norm = wav / np.max(np.abs(wav)) * level
    return wav_norm


wav_norm0 = normalise(wav_short)
wav_norm3 = normalise(wav_short, -3)
wav_norm18 = normalise(wav_short, -18)

# %% move functuion to dedicated utility module
import dap
wav_norm = dap.normalise(wav_short, -60)
