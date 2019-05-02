#!/usr/bin/env python3
#
# Copyright 2019 by Branislav Gerazov
#
# See the file LICENSE for the license associated with this software.
#
# Author(s):
#   Branislav Gerazov, May 2019

"""
Digital Audio Systems

Excercise 05: Digital Audio Effects: Robotiser.

@author: Branislav Gerazov
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal as sig
from scipy import fftpack as fft
import os
import das

# %% load wave
path = 'audio/'
file_name = 'Donco22k.wav'
os.system('play '+path+file_name)
fs, wav = wavfile.read(path+file_name)
wav = wav / 2**15
t = np.arange(wav.size) / fs

# %% window + robot
n_win = 512
win = sig.get_window('hann', n_win)
t_win = n_win / fs
n_half = n_win // 2
n_hop = n_half
n_pad = n_half
pad = np.zeros(n_pad)
wav_pad = np.concatenate((pad, wav, pad))

pos = 0
wav_robot = np.zeros(wav_pad.size)
while pos <= wav_pad.size - n_win:
    frame = wav_pad[pos : pos+n_win]
    frame = frame * win
    
    frame_fft = fft.fft(frame, n_win)
    frame_amp = np.abs(frame_fft)
    frame_ifft = fft.ifft(frame_amp)
    frame_robot = frame_ifft.real
    frame_robot = fft.fftshift(frame_robot)
    
    wav_robot[pos : pos+n_win] += frame_robot * win
    
    pos += n_hop

# %% wav write and play
wav_robot_int16 = wav_robot * 2**15
wav_robot_int16 = wav_robot_int16.astype('int16')
wavfile.write(path+file_name+'_robot.wav', fs, wav_robot_int16)
os.system('play '+path+file_name+'_robot.wav')
