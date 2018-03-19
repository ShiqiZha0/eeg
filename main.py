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

raw_data = edfparser.main('test.edf')
pre_data_list = [[0 for col in range(128)] for row in range(14)]
for i in range(len(raw_data)):
    if (i >= 2 and i <= 15):
        #for j in range(len (raw_data[i])):
        for j in range(128):
            pre_data_list[i-2][j] = raw_data[i][j]

pre_data = np.asarray(pre_data_list)

plt.plot(pre_data,'o')
         
filter_data = filter.main(pre_data)

#plt.plot(filter_data,'p')
print (filter_data.shape)
fft_data =  pyfftw.FFTW(filter_data)
plt.plot(fft_data,'o')
#fffffffffffff
