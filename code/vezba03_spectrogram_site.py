# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
#from math import pi
from scipy.io import wavfile
#import os
from scipy import fftpack as fp 
from scipy import signal as sig 
import das

fs, wav = wavfile.read('audio/zvona2.wav')
#os.system('play audio/wav.wav')
wav = wav / 2**15
t = np.arange(0, wav.size/fs, 1/fs)
f, wav_amp = das.get_spectrum(wav, fs)

#%% aliasing
#fs = fs/4
#wav = wav[0:-1:4]
#wavfile.write('audio/Zvona_alias.wav', fs, wav)
# definiranje prozorec
#M = wav.size
#Nt = .050  # ms
#N = np.round(Nt * fs)
plt.figure(figsize=(16,4))
for i, N in enumerate([128, 512, 4096, 4096*2]):# 16384]):
#for i, wintype in enumerate(['boxcar', 'hann','hamming','blackmanharris']):
#    N = 2048  # 2**11
    Nh = int(N/2)
    H = int(N/2)  # hop size
#    print N, Nh
    win = sig.get_window('hamming',N)
#    win = sig.get_window(wintype,N)    
    poz = Nh  # pozicija na sredinata na prozorecot
    pad = np.zeros(Nh)
    wav_pad = np.concatenate((pad, wav, pad))
    M = wav.size + N  # dolzina na wav_pad
    while poz < M-Nh:
        frame = wav_pad[poz-Nh : poz+Nh] * win
        f_frame, frame_spec = das.get_spectrum(frame, fs)
        if poz == Nh:
            frames_spec = np.array([frame_spec]).T
        else:
            frames_spec = np.hstack((frames_spec,\
                          np.array([frame_spec]).T))
        poz += H                      
        
    #plt.figure()
    #plt.subplot(211)
    #plt.plot(t, wav)
    #plt.subplot(212)
    #plt.plot(f, wav_amp)
    #%%
    #from colormaps import cmaps
    #plt.register_cmap(name='viridis', cmap=cmaps.viridis)
    #plt.set_cmap(cmaps.viridis)

    plt.subplot(1,4,i+1)
    plt.imshow(frames_spec, aspect='auto', \
               origin='lower', \
               extent=[0, t[-1], 0, f_frame[-1]],\
               vmin=-100,vmax=0, cmap='viridis')
#    if i == 1 or i == 3:
    if i > 0:
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
    else:
        plt.ylabel('Frequency [Hz]')
    
#    if i <= 1:
#        plt.gca().xaxis.set_major_locator(plt.NullLocator())
#    else:
    plt.xlabel('Time [s]')
        #cbar = plt.colorbar()
    #plt.xlabel('time [s]')
    #plt.ylabel('frequency [Hz]')
    #cbar.ax.set_ylabel('Amplitude [dB]')
               
    #plt.axis([0.5, 1.5, 0, 2500])
    #plt.axis([0, t[-1], 0, 5500])
    plt.axis([0, t[-1], 0, 10000])
#    plt.axis([0, 5, 1000, 4000])
    #plt.axis([0, t[-1], 1000, 1500])
    
    ##%%
    #import matplotlib.image as img
    #lena = img.imread('lena.jpg')
    #lena_mono = np.sum(lena,2)/3
    #plt.figure()
    #plt.imshow(lena_mono, cmap='jet')
    #plt.colorbar()
    #
    ##%%
    #lisa = img.imread('lisa.jpg')
    #lisa_mono = np.sum(lisa,2)/3
    #plt.figure()
    #plt.imshow(lisa_mono, cmap='gray')
    #plt.colorbar()
plt.tight_layout()
plt.show()