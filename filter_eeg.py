# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 10:13:44 2018

@author: shiqizhao
"""

import filter
import edfparser
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.fftpack import fft,ifft
import FFT
import math
import numpy as np
import scipy.signal as signal
from sklearn import preprocessing

'''
t = []
a = [2000,2002,2010,2005,2007,2011]
X_scaled = preprocessing.scale(a)
x = fft(X_scaled)
b = fft (a)
print (b)


for i in range(6):
    t.append(i)
    
plt.plot(t,abs(x))

'''
f1=0.5
f2=0.9
f3=1.1
f4=1.5
N = 256
t = []
#N=length(t)
fs=128
M=512
raw_data = edfparser.main('data\zsq-right.edf')
#print (raw_data.shape)
pre_data_list = [[0 for col in range(128)] for row in range(14)]
for i in range(len(raw_data)):
    if (i >= 2 and i <= 15):
        #for j in range(len (raw_data[i])):
        for j in range(128):
            pre_data_list[i-2][j] = raw_data[i][j]
            
pre_data = np.asarray(pre_data_list)
print (pre_data.shape)


for i in range(128):
    t.append(i)
    
    
    
x1 = preprocessing.scale(pre_data[6])



plt.subplot(221)
plt.plot(t,x1)
y=fft(x1)

xf = np.linspace(0,64,num=N/4)
#xf = 100*xf

plt.subplot(222)
plt.plot(xf,abs(y[0:int(N/4)]))
#plt.plot(xf,y[0:int(N/2)])
'''
wc1=2*f2/fs
wc2=2*f3/fs
wc3=2*f4/fs
A=[0,1,0]
weigh=[1,1,1]
'''

bands = [0,9.99,10,30,30.01,64]
desired = [0, 0, 1, 1, 0, 0]
fir_firls = signal.firwin2(60, bands, desired, fs=fs)

freq, response = signal.freqz(fir_firls)

xxf = np.linspace(0,64,num=M)
plt.subplot(223)
plt.plot(xxf,abs(np.array(response)))

x2=signal.lfilter(fir_firls,1,x1)

S1=fft(x2)

xxxf = np.linspace(0,64,num=64)

plt.subplot(224)
plt.plot(xxxf,abs(S1[0:64]))
