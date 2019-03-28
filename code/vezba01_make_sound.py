#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Copyright 2017 - 2019 by Branislav Gerazov
#
# See the file LICENSE for the license associated with this software.
#
# Author(s):
#   Branislav Gerazov, March 2017

"""
Digital Audio Systems

Excercise 01: Make sound.

@author: Branislav Gerazov
"""
from __future__ import division
import numpy as np
from scipy.io import wavfile
import os
from math import pi

# %% input defined using terminal
# print 'sys.argv : ', sys.argv, 'size is ', len(sys.argv)
# f = int(sys.argv[1])

# %% input defined in script
f = 200

# %% generate sound
fs = 44100
t = np.arange(0, 2, 1/fs)
sine = np.sin(2*pi*f*t)

# %% save and play
wavfile.write('audio/sinus.wav', fs,
              np.array(sine * 2**15, dtype='int16'))
os.system('play audio/sinus.wav')

# %%
import das
das.get_sound(fs=48000, f=20000)
