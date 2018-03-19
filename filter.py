# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 11:01:27 2018

@author: shiqizhao
"""

# utils.py
from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
import soundfile as sf

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def main(raw_data):
    # Read input wav file
    data = raw_data
    #data, fs = sf.read('david1.wav')
    fs = 128

    Time0 = np.linspace(0, len(data)/fs, num=len(data))

    # Sample rate and desired cutoff frequencies (in Hz).
    lowcut = 10
    highcut = 30

    #apply bandpass filter
    y = butter_bandpass_filter(data, lowcut, highcut, fs, order=6)
    return y
    #output filtered wav file 
    #sf.write('entry.wav', y, fs)