#!/usr/bin/env python3
#
# Copyright by Branislav Gerazov 2019 - 2020
#
# See the file LICENSE for the license associated with this software.
#
# Author(s):
#   Branislav Gerazov, May 2019 - 2020

"""
Digital Audio Processing

Excercise 05: Digital Audio Effects: Tremolo.

@author: Branislav Gerazov
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import dap

# %% load wave
path = 'audio/'
file_name = 'Mara.wav'
os.system('play ' + path + file_name)
fs, wav = wavfile.read(path + file_name)
t = np.arange(wav.size) / fs

# %% generate tremolo sine
amp = 0.5
offset = 1
f = 50
tremolo = amp * np.sin(2*np.pi*f*t) + offset
plt.plot(t, tremolo)

# %% apply tremolo
wav_tremolo = wav * tremolo
wav_tremolo = dap.normalise(wav_tremolo, -3)

# %% wav write and play
wav_tremolo_int16 = wav_tremolo * 2**15
wav_tremolo_int16 = wav_tremolo_int16.astype('int16')
wavfile.write(path + file_name + '_tremolo.wav', fs, wav_tremolo_int16)
os.system('play ' + path + file_name + '_tremolo.wav')


