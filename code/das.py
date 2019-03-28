#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2017 - 2019 by Branislav Gerazov
#
# See the file LICENSE for the license associated with this software.
#
# Author(s):
#   Branislav Gerazov, Mar 2017 - 2019

"""
Digital Audio Systems

Utility functions.

@author: Branislav Gerazov
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy import fftpack as fft
from scipy import signal as sig
import os


def normalise(wav, level_dB=0):
    """
    Normalise WAV file.

    Parameters
    ----------
    wav : ndarray
        Input audio.
    level_dB : int
        Level in dB.

    Returns
    ----------
    wav_norm : ndarray
        Normalised audio.
    """
    level = 10**(level_dB/20)
    wav_norm = wav / np.max(np.abs(wav)) * level
    return wav_norm


def get_sound(fs=16000, t=2, f=200, level=-3, plot=False, play=True):
    """
    Generate sine tone.

    Parameters
    ----------
    fs : int
        Sampling rate.
    t : int
        Length in seconds.
    f : int
        Frequency of sine tone.
    level : int/float
        Level in dB.
    plot : bool
        Plot the sine tone.
    play : bool
        Play the sine tone.

    Returns
    ----------
    wav : ndarray
        Generated sine tone.
    """
    t = np.arange(0, t, 1/fs)
    wav = np.sin(2*np.pi*f*t)
    wav = normalise(wav, level)
    if plot:
        plt.figure()
        plt.plot(t, wav)
        plt.grid()
    if play:
        wavfile.write('audio/sine.wav', fs,
                      np.array(wav * 2**15, dtype='int16'))
        os.system('play audio/sine.wav')
    return wav


def get_spectrum(fs, wav, n_fft=None, log=True, plot=False):
    """
    Calculate spectrum of signal.

    Parameters
    ----------
    fs : int
        Sampling rate.
    wav : ndarray
        Input audio.
    n_fft : int
        Number of FFT bins to use.
    plot: bool
        Plot the spectrum.

    Returns
    ----------
    f : ndarray
        Frequency of FFT bins in Hz.

    wav_amp : ndarray
        Amplitude specrum.
    """
    n = wav.size
    if n_fft is None:
        n_fft = np.ceil(np.log2(n))
        n_fft = int(2**n_fft)
    n_keep = int(n_fft/2) + 1
    wav_fft = fft.fft(wav, n_fft)
    wav_amp = np.abs(wav_fft)
    wav_amp = wav_amp / n
    wav_amp = wav_amp[:n_keep]
    wav_amp[1:-1] = 2*wav_amp[1:-1]
    if log:
        eps = np.finfo(float).eps
        wav_amp[wav_amp < eps] = eps
        wav_amp = 20*np.log10(wav_amp)
    f = np.linspace(0, fs/2, n_keep)
    if plot:
        plt.figure()
        plt.plot(f, wav_amp)
        plt.axis([0, fs/2, -100, 0])
        plt.grid()
    return f, wav_amp


def get_spectrogram(fs, wav, n_win=2048, win_type='hamming',
                    spec=True, plot=True):
    """
    Calculate STFT spectrogram of signal.

    Parameters
    ----------
    fs : int
        Sampling rate.
    wav : ndarray
        Input audio.
    n_win : int
        Window length used in analysis.
    win_type : str
        Window type to use.
    plot : bool
        Plot the spectrogram.

    Returns
    ----------
    t_frames : ndarray
        Time locations of frame centers.
    f_frame : ndarray
        Frequency of FFT bins in Hz.
    frames : ndarray
        Frames extracted from audio file or spectrogram in dB based on spec.
    """
    win = sig.get_window(win_type, n_win)
    n_half = n_win // 2
    n_hop = n_half
    pad = np.zeros(n_half)
    wav_pad = np.concatenate((pad, wav, pad))
    pos = 0
    while pos <= wav_pad.size - n_win:
        frame = wav_pad[pos: pos+n_win]
        frame = frame * win
        if spec:
            f_frame, frame = get_spectrum(fs, frame, n_fft=n_win)
        else:
            f_frame = np.arange(frame.size)
        frame = frame[:, np.newaxis]
        if pos == 0:
            frames = frame
        else:
            frames = np.concatenate((frames, frame), axis=1)
        pos += n_hop
    n_frame = frames.shape[1]
    t_frame = np.arange(n_frame) * n_hop/fs
    if plot:
        plt.figure()
        plt.imshow(frames, aspect='auto', origin='lower',
                   extent=[0, t_frame[-1], 0, f_frame[-1]],
                   vmin=-100, vmax=100)
        plt.colorbar()
    return t_frame, f_frame, frames


def overlapadd(frames):
    """
    Overlap and add frames. Overlap is fixed at 50%.

    Parameters
    ----------
    frames : ndarray
        Frames of signal.

    Returns
    ----------
    wav : ndarray
        Output signal.
    """
    win_s = frames.shape[0]
    no_frames = frames.shape[1]
    H = int(win_s / 2)
    wav = np.zeros((no_frames+1)*H)
    pos = 0
    for frame_filt in frames.T:
        wav[pos:pos+win_s] = wav[pos:pos+win_s] + frame_filt
        pos += H

    return wav
