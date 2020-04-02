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

Excercise 04: FIR and IIR filter demo.

@author: Branislav Gerazov
"""
from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
import os
from scipy import signal as sig
import dap

# %% load wav
folder = 'audio/'
filename = 'Mara.wav'
fs, wav = wavfile.read(folder+filename)
wav = wav / 2**15
t = np.arange(wav.shape[0]) / fs
os.system('play '+folder+filename)
plt.plot(t, wav)
dap.get_spectrogram(fs, wav)

# %% FIR filters
# low pass
f_l = 150
orders = [5, 11, 51, 101, 501]
plt.figure()
leg = []
for order in orders:
    b_fir = sig.firwin(order, f_l,
                       window='hamming',
                       pass_zero=True,
                       nyq=fs/2)
    w, h_spec = sig.freqz(b_fir, 1)
    plt.subplot(1, 2, 1)
    plt.plot(w/np.pi*fs/2,
             20*np.log10(np.abs(h_spec)))
    plt.grid('on')
    plt.subplot(1, 2, 2)
    plt.plot(b_fir)
    plt.grid('on')
    leg.append(order)
plt.legend(leg)

# %% filter signal
b_fir = sig.firwin(501, f_l,
                   window='hamming', pass_zero=True, nyq=fs/2)
wav_lp_fir = sig.lfilter(b_fir, 1, wav)
wav_lp_filtfilt = sig.filtfilt(b_fir, 1, wav)

# %% plot
plt.figure()
plt.plot(t, wav)
plt.plot(t, wav_lp_fir)
plt.plot(t, wav_lp_filtfilt)

# %% plot spectrums
f, wav_spec = dap.get_spectrum(fs, wav)
f, wav_lp_spec = dap.get_spectrum(fs, wav_lp_fir)
f, wav_lp_spec_filtfilt = dap.get_spectrum(fs, wav_lp_filtfilt)

plt.figure()
plt.plot(f, wav_spec)
plt.plot(f, wav_lp_spec)
plt.plot(f, wav_lp_spec_filtfilt)
plt.xscale('log')
plt.axis((20, 16000, -80, 0))
plt.grid()

# %%
orders = [3, 5, 7, 11]
leg = []
plt.figure()
for order in orders:
    b_iir, a_iir = sig.iirfilter(order, np.array(f_l/(fs/2)),
                                 btype='lowpass', rp=3, rs=3, ftype='cheby1')

    w, h_spec = sig.freqz(b_iir, a_iir)
    plt.subplot(1, 2, 1)
    plt.plot(w/np.pi*fs/2, 20*np.log10(np.abs(h_spec)))
    plt.grid('on')

    # find impulse response
    excitation = np.zeros(500)
    excitation[0] = 1
    h_imp = sig.lfilter(b_iir, a_iir, excitation)
    plt.subplot(1, 2, 2)
    plt.plot(h_imp)
    plt.grid('on')
    leg.append(order)

plt.legend(leg)

# %% filter signal
b_iir, a_iir = sig.iirfilter(5, np.array(f_l/(fs/2)),
                             btype='lowpass', ftype='butter')
wav_lp_iir = sig.lfilter(b_iir, a_iir, wav)
wav_lp_iir_filtfilt = sig.filtfilt(b_iir, a_iir, wav)

# %% plot
plt.figure()
plt.plot(t, wav)
plt.plot(t, wav_lp_fir)
plt.plot(t, wav_lp_iir)
plt.plot(t, wav_lp_iir_filtfilt)

# %% plot spectrums
f, wav_spec = dap.get_spectrum(fs, wav)
f, wav_lp_spec_iir = dap.get_spectrum(fs, wav_lp_iir)
f, wav_lp_spec_iir_filtfilt = \
            dap.get_spectrum(fs, wav_lp_iir_filtfilt)
plt.figure()
plt.plot(f, wav_spec)
plt.plot(f, wav_lp_spec, alpha=.7)
plt.plot(f, wav_lp_spec_iir, alpha=.7)
plt.xscale('log')
plt.axis((20, 16000, -80, 0))
plt.grid()

# %% play
wavfile.write('audio/Mara_lp_fir.wav', fs,
              np.array(wav_lp_fir*2**15,
                       dtype='int16'))
os.system('play audio/Mara_lp_fir.wav')
