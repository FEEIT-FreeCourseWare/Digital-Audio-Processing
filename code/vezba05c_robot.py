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

Excercise 05: Digital Audio Effects: Robotiser.

@author: Branislav Gerazov
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal as sig
from scipy import fftpack as fp
import os

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
    
    frame_fft = fp.fft(frame, n_win)
    frame_amp = np.abs(frame_fft)
    frame_ifft = fp.ifft(frame_amp)
    frame_robot = frame_ifft.real
    frame_robot = fp.fftshift(frame_robot)
    
    wav_robot[pos : pos+n_win] += frame_robot * win
    
    pos += n_hop

# %% wav write and play
wav_robot_int16 = wav_robot * 2**15
wav_robot_int16 = wav_robot_int16.astype('int16')
wavfile.write(path+file_name+'_robot.wav', fs, wav_robot_int16)
os.system('play '+path+file_name+'_robot.wav')
