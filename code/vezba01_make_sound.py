#!/usr/bin/env python3
#
# Copyright by Branislav Gerazov 2017 - 2020
#
# See the file LICENSE for the license associated with this software.
#
# Author(s):
#   Branislav Gerazov, March 2017 - 2020

"""
Digital Audio Processing

Excercise 01: Make sound.

@author: Branislav Gerazov
"""
import numpy as np
from scipy.io import wavfile
import os

# %% input defined using terminal
# print 'sys.argv : ', sys.argv, 'size is ', len(sys.argv)
# f = int(sys.argv[1])

# %% input defined in script
f = 200

# %% generate sound
fs = 44100
t = np.arange(0, 2, 1/fs)
sine = np.sin(2 * np.pi * f * t)

# %% save and play
wavfile.write(
    'audio/sinus.wav', fs, np.int16(sine * 2**15)
    )
os.system('play audio/sinus.wav')

# %% port function to dap module
import dap
dap.get_sound(fs=48000, f=20000)
