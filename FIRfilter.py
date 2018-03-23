# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 20:58:00 2018

@author: shiqizhao
"""

import math
import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import scipy.signal as signal

f1=0.5
f2=0.9
f3=1.1
f4=1.5
N = 1023
t = []
#N=length(t)
fs=10
M=512
for i in range(1023):
    t.append(i)
x1=[math.sin(2*math.pi*(f1/fs)*t)+math.sin(2*math.pi*(f2/fs)*t)+math.sin(2*math.pi*(f3/fs)*t)+math.sin(2*math.pi*(f4/fs)*t) for t in range(1023)]

plt.subplot(221)
plt.plot(t,x1)
y=fft(x1)
xf = np.linspace(0,1/2-1/N,num=N/2)
xf = 10*xf

plt.subplot(222)
plt.plot(xf,abs(y[0:int(N/2)]))
wc1=2*f2/fs
wc2=2*f3/fs
wc3=2*f4/fs
A=[0,1,0]
weigh=[1,1,1]
bands = [0,0.05,0.7,1.3,1.75,5]
desired = [0, 0, 1, 1, 0, 0]
fir_firls = signal.firwin2(60, bands, desired, fs=fs)

freq, response = signal.freqz(fir_firls)
print (response.shape)
plt.plot(xf,abs(y[0:int(N/2)]))
xxf = np.linspace(0,1-1/M,num=M)
xxf = xxf*5
plt.subplot(223)
plt.plot(xxf,abs(np.array(response)))



#plt.plot(fft_pre_x,fft_pre_y,'b')
#plt.subplot(122)

'''

import numpy as np  
import matplotlib.pyplot as pl  
import matplotlib  
import math  
import random  
  
row = 4  
col = 4  
  
N = 500  
fs = 5  
n = [2*math.pi*fs*t/N for t in range(N)]    # 生成了500个介于0.0-31.35之间的点  
# print n  
axis_x = np.linspace(0,3,num=N)  
  
#频率为5Hz的正弦信号  
x = [math.sin(i) for i in n]  
pl.subplot(221)  
pl.plot(axis_x,x)  
pl.title(u'5Hz的正弦信号')  
pl.axis('tight')  
  
#频率为5Hz、幅值为3的正弦+噪声  
x1 = [random.gauss(0,0.5) for i in range(N)]  
xx = []  
#有没有直接两个列表对应项相加的方式？？  
for i in range(len(x)):  
    xx.append(x[i]*3 + x1[i])  
  
pl.subplot(222)  
pl.plot(axis_x,xx)  
pl.title(u'频率为5Hz、幅值为3的正弦+噪声')  
pl.axis('tight')  
  
#频谱绘制  
xf = np.fft.fft(x)  
xf_abs = np.fft.fftshift(abs(xf))  
axis_xf = np.linspace(-N/2,N/2-1,num=N)  
pl.subplot(223)  
pl.title(u'频率为5Hz的正弦频谱图')  
pl.plot(axis_xf,xf_abs)  
pl.axis('tight')  
  
#频谱绘制  
xf = np.fft.fft(xx)  
xf_abs = np.fft.fftshift(abs(xf))  
pl.subplot(224)  
pl.title(u'频率为5Hz的正弦频谱图')  
pl.plot(axis_xf,xf_abs)  
pl.axis('tight')  
  
pl.show()  




figure(1);
subplot(211);plot(t,x1);title('原信号');
y=fft(x1);
f=(0:1/N:1/2-1/N)*fs;
subplot(212);plot(f,abs(y(1:N/2)));grid;xlabel('hz');%处理前频谱
wc1=2*f2/fs;wc2=2*f3/fs;wc3=2*f4/fs;%归一化角频率，用于下面的f1
f1=[0 wc1-0.05 wc1 wc2 wc2+0.05 1];
A=[0 0 1 1 0 0];%设置带通或带阻，1为带通，0为带阻	
weigh=[1 1 1 ];%设置通带和阻带的权重
b=remez(60,f1,A,weigh);%传函分子
h1=freqz(b,1,M);%幅频特性
figure(2)
f=(0:1/M:1-1/M)*fs/2;
subplot(211);plot(f,abs(h1));grid;title('带通');
x2=filter(b,1,x1);
S1=fft(x2);
f=(0:1/N:1/2-1/N)*fs;
subplot(212);plot(f,abs(S1(1:N/2)));grid;xlabel('hz');%处理后频谱
'''