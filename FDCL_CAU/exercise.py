import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import torch
import librosa.display
import librosa
from scipy import signal
import glob

Fs = 25600 #샘플링 프리퀀시

# 하나만 테스트용  >> freq 10000 근처에서 잡힘
sign1 = scipy.io.loadmat('D:/DSdata/A/data/MB_H.mat')
sign1 = np.ravel(sign1['Signal'],order='C')
# sign1 = sign1[0:131073]
#
#
# asignal = librosa.stft(sign1, n_fft=2048, hop_length = 128, win_length=512)
# spec = np.abs(asignal)
# librosa.display.specshow(spec, sr=12800, hop_length=128)
# plt.show()


# plt.plot(sign1)
# plt.show()
#
# f,tt,sxx = signal.spectrogram(sign1,fs=12800, scaling='spectrum')
# # f = sampling freq, tt = segment times, sxx= spectrogram of x
# plt.pcolormesh(tt,f,sxx, shading ='gouraud')
# # plt.title('Spectrogram')
# # plt.ylabel('Frequency [Hz]')
# # plt.xlabel('Time [sec], 6second per day, 50 days in total')
# plt.axis('off')
# plt.show()
# # plt.savefig('a.png')

# # 다 불러오는 버젼
# sign = []
# for filename in glob.glob('D:/DSdata/A/data/*.mat'):  # mat 파일 불러오기
#     mat = scipy.io.loadmat(filename)
#     mat = np.ravel(mat['Signal'],order='C') #진동 데이터 1-d array 변환
#     sign = np.append(sign,mat[0,:])
#
#
# #주파수 데이터 시각화
# # plt.rcParams["figure.figsize"] = (15,4)
# plt.plot(sign)
# plt.xlabel('Time [s]')
# plt.ylabel('Acceleration [g]')
# plt.show()
#
# #스펙트로그램 시각화
# f,tt,sxx = signal.spectrogram(sign1,Fs)
# plt.pcolormesh(tt,f,sxx, shading ='gouraud')
# plt.title('Spectrogram')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec], 6second per day, 50 days in total')
# plt.show()


jb = scipy.io.loadmat('D:/DSdata/JB/A_even.mat')
print(jb)
