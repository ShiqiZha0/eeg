# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 10:14:22 2018

@author: shiqizhao
"""

import filter
import edfparser
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.fftpack import fft,ifft
import FFT

raw_data = edfparser.main('data\lkf-right.edf')
#print (raw_data.shape)
pre_data_list = [[0 for col in range(128)] for row in range(14)]
for i in range(len(raw_data)):
    if (i >= 2 and i <= 15):
        #for j in range(len (raw_data[i])):
        for j in range(128):
            pre_data_list[i-2][j] = raw_data[i][j]

pre_data = np.asarray(pre_data_list)
print (pre_data.shape)
#plt.subplot(221)
#plt.plot(pre_data,'o')

fft_pre_x,fft_pre_y= FFT.fft_transform(pre_data[10])

#plt.subplot(222)
plt.plot(fft_pre_x,fft_pre_y,'b')


         
filter_data = filter.main(pre_data)

#plt.plot(filter_data,'p')
#print(type(filter_data))
print (filter_data.shape)
#fft_data =  pyfftw.FFTW(filter_data)
#plt.plot(fft_data,'o')
