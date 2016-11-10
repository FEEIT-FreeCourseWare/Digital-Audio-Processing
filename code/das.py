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

Utility functions.

@author: Branislav Gerazov
"""
from __future__ import division
import numpy as np
import scipy.fftpack as ffp

#%% functions
def get_spectrum(wav, fs):
    N = wav.size
    Nfft = 2**np.ceil(np.log(N)/np.log(2))
    wav_spec = ffp.fft(wav, Nfft)
    wav_amp = np.abs(wav_spec)
    wav_amp = wav_amp[0:Nfft/2+1]
    wav_amp = wav_amp / N
    wav_amp[1:-1] = wav_amp[1:-1] * 2
    wav_amp = 20*np.log10(wav_amp)
    f = np.linspace(0, fs/2, Nfft/2+1)
    return f, wav_amp
    