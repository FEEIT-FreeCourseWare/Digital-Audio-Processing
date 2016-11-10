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

Excercise 01: Make sound.

@author: Branislav Gerazov
"""
from __future__ import division
import numpy as np
from scipy.io import wavfile
import os
from math import pi
import sys

#%% generate sine tone
print 'sys.argv : ', sys.argv, 'size is ', len(sys.argv)
f = int(sys.argv[1])
fs = 44100
t = np.arange(0, 2, 1/fs)
sine = np.sin(2*pi*f*t)

#%% save to WAV file
wavfile.write('sinus.wav',fs,np.array(sine * 2**15, dtype='int16'))

#%% play file
os.system('play sinus.wav')