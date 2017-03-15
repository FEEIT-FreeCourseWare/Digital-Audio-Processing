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

Utility functions.

@author: Branislav Gerazov
"""
from __future__ import division
import numpy as np
from matplotlib import pyplot as plt 
from scipy.io import wavfile
import os
from scipy import fftpack as fftp
from scipy import signal as sig

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

def get_sound(fs=16000, T=2, f=200, level=-3,
              plot=False, play=True):
    """
    Generate sine tone.
    
    Parameters
    ----------
    fs : int
        Sampling rate.
    T : int
        Length in seconds.
    f : int
        Frequency of sine tone.
    level : int/float
        Level in dB.
    plot: bool
        Plot the sine tone.
    play : bool
        Play the sine tone.
        
    Returns 
    ----------
    
    zvuk : ndarray
        Generated sine tone.        
    """ 
    t = np.arange(0,T,1/fs)
    zvuk = np.sin(2*np.pi*f*t)
    zvuk = normalise(zvuk, level)

    if plot:
        plt.figure()
        plt.plot(t,zvuk)
        plt.grid()
    
    if play:
        wavfile.write('audio/sinus.wav', fs, 
                      np.array(zvuk*2**15, dtype='int16'))
        os.system('play audio/sinus.wav')
    
    return zvuk

def get_spectrum(fs, wav, Nfft=None, plot=False):
    """
    Calculate spectrum of signal.
    
    Parameters
    ----------
    fs : int
        Sampling rate.
    wav : ndarray
        Input audio.
    Nfft : int
        Number of FFT bins to use.
    plot: bool
        Plot the spectrum.
        
    Returns 
    ----------
    f : ndarray
        Frequency of FFT bins in Hz.
    
    wav_amp : ndarray
        Amplitude specrum in dB.   
    """
    N = wav.size
    if Nfft is None:
        Nfft = int(2**np.ceil(np.log(N)/np.log(2)))
    wav_fft = fftp.fft(wav, Nfft)
    wav_fft = wav_fft / N
    wav_fft = wav_fft[0:int(Nfft/2+1)]
    wav_fft[1:int(Nfft/2)] = 2 * wav_fft[1:int(Nfft/2)] 
    wav_amp = np.abs(wav_fft)
    eps = np.finfo(float).eps
    wav_amp[wav_amp < eps] = eps
    wav_amp = 20*np.log10(wav_amp)
    wav_ph = np.angle(wav_fft)
    wav_ph = np.unwrap(wav_ph)
    f = np.linspace(0, fs/2, Nfft/2+1)
    if plot:
        plt.figure()
        plt.plot(f, wav_amp)
        plt.xscale('log')
        plt.axis((20, fs/2, -100, 0))
        plt.grid()
        
    return f, wav_amp
    
def get_spectrogram(fs, wav, N=2048, win='hamming', plot=True):
    """
    Calculate STFT spectrogram of signal.
    
    Parameters
    ----------
    fs : int
        Sampling rate.
    wav : ndarray
        Input audio.
    N : int
        Window length used in analysis.
    win : str
        Window type to use.
    plot: bool
        Plot the spectrogram.
        
    Returns 
    ----------
    t_frames : ndarray
        Time locations of frame centers. 
    f_frame : ndarray
        Frequency of FFT bins in Hz.
    frame_specs : ndarray
        Spectrogram in dB.
    frames : ndarray
        Frames extracted from audio file.
    """
    Nh = int(N/2)
    H = Nh
    w = sig.get_window(win, N)
    pad = np.zeros(Nh)
    wav_pad = np.r_[pad, wav, pad]
    poz = 0
    while poz <= wav_pad.size - N:
        frame = wav_pad[poz:poz+N]
        frame = frame * w
        f_frame, frame_spec = get_spectrum(fs, frame, N)
        if poz == 0:
            frames =frame[:,np.newaxis]
            frame_specs =frame_spec[:,np.newaxis]
        else:
            frames = np.hstack((frames, frame[:,np.newaxis]))
            frame_specs = np.hstack((frame_specs, 
                                     frame_spec[:,np.newaxis]))
        poz += H
        
    no_frames = frame_specs.shape[1]
    t_frames = np.arange(no_frames) * H / fs
    if plot:
        plt.figure()
        plt.imshow(frame_specs, 
                   extent=[0, t_frames[-1], 0, f_frame[-1]],
                   aspect='auto',
                   origin='lower',
                   vmin=-100,
                   vmax=0,
                   cmap='viridis')
        
    return t_frames, f_frame, frame_specs, frames

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
    win_s = frames_filt.shape[0]
    no_frames = frames_filt.shape[1]
    H = int(win_s / 2)
    wav = np.zeros((no_frames+1)*H)
    pos = 0
    for frame_filt in frames_filt.T:
        wav[pos:pos+win_s] = wav[pos:pos+win_s] + frame_filt
        pos += H
        
    return wav_filt
