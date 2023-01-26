# import torchaudio.transforms as AT
import librosa.filters as F
import scipy.io
import os
from scipy import signal
import glob
import numpy as np
import math
import matplotlib.pyplot as plt

filepath = 'D:/DSdata/A/spec/'
for filename in glob.glob('D:/DSdata/A/data/*.mat'):
    temp_mat = scipy.io.loadmat(filename)
    mat = temp_mat['Signal']
    file = filename.split('\\')[-1]
    file = file.split('.')[0]
    for i in range(765):
        f,tt,sxx = signal.spectrogram(mat[i],fs=12800)
        plt.pcolormesh(tt,f,sxx, shading='gouraud')
        plt.axis('off')

        imagename = "/{:d}.jpg".format(i)
        dir = filepath + file
        if not os.path.isdir(dir):
            os.makedirs(dir)

        plt.tight_layout()
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
        plt.savefig(dir+imagename)

# for i in filelist:
#     aa = scipy.io.loadmat(path + i)
#     temp_signal = aa['Signal']
#     temp_signal2 = np.transpose(temp_signal)
#     temp_list = [path + i, temp_signal2]
#     signal_list.append(temp_list)


# mb_vmat=signal_list[11]
# for k in  range(267,753):
#     sig = mb_vmat[1][k,:]
#
#
#     f,tt,sxx = signal.spectrogram(sig,fs=12800)
#     plt.pcolormesh(tt,f,sxx, shading='gouraud')
#     plt.axis('off')
#
#
#     filepath ="C:/Users/ksk/Desktop/hyunmin/A호기/MB_V.MAT/"
#     filename= f'MB_Vmat{k}.png'
#     plt.tight_layout()
#     plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
#     plt.savefig(filepath+filename)


# mb_vmat1=signal_list[0]
# for k in  range(765):
#     sig = mb_vmat1[1][k,:]


#     f,tt,sxx = signal.spectrogram(sig,fs=12800)
#     plt.pcolormesh(tt,f,sxx, shading='gouraud')
#     plt.axis('off')


#     filepath ="C:/Users/ksk/Desktop/hyunmin/A호기/GB_3_MP_GS_R.MAT/"
#     filename= f'GB_3_MP_GS_RMAT{k}.png'
#     plt.tight_layout()
#     plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
#     plt.savefig(filepath+filename)

# mb_vmat2=signal_list[1]
# for k in  range(643,765):
#     sig = mb_vmat2[1][k,:]


#     f,tt,sxx = signal.spectrogram(sig,fs=12800)
#     plt.pcolormesh(tt,f,sxx, shading='gouraud')
#     plt.axis('off')


#     filepath ="C:/Users/ksk/Desktop/hyunmin/A호기/GB_3_PB_GS_H.MAT/"
#     filename= f'GB_3_PB_GS_HMAT{k}.png'
#     plt.tight_layout()
#     plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
#     plt.savefig(filepath+filename)

# print("adada")
#
# mb_vmat4=signal_list[3]
# for k in  range(765):
#     sig = mb_vmat4[1][k,:]


#     f,tt,sxx = signal.spectrogram(sig,fs=12800)
#     plt.pcolormesh(tt,f,sxx, shading='gouraud')
#     plt.axis('off')


#     filepath ="C:/Users/ksk/Desktop/hyunmin/A호기/GB_3_RS_H.MAT/"
#     filename= f'GB_3_RS_HMAT{k}.png'
#     plt.tight_layout()
#     plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
#     plt.savefig(filepath+filename)

# print("ada")


# mb_vmat5=signal_list[4]
# for k in  range(765):
#     sig = mb_vmat5[1][k,:]


#     f,tt,sxx = signal.spectrogram(sig,fs=12800)
#     plt.pcolormesh(tt,f,sxx, shading='gouraud')
#     plt.axis('off')


#     filepath ="C:/Users/ksk/Desktop/hyunmin/A호기/GB_3_WB_GS_R.mat/"
#     filename= f'GB_3_WB_GS_RMAT{k}.png'
#     plt.tight_layout()
#     plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
#     plt.savefig(filepath+filename)

# mb_vmat6=signal_list[5]
# for k in  range(765):
#     sig = mb_vmat6[1][k,:]


#     f,tt,sxx = signal.spectrogram(sig,fs=12800)
#     plt.pcolormesh(tt,f,sxx, shading='gouraud')
#     plt.axis('off')


#     filepath ="C:/Users/ksk/Desktop/hyunmin/A호기/GB_SP_LHS_V.mat/"
#     filename= f'GB_SP_LHS_VMAT{k}.png'
#     plt.tight_layout()
#     plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
#     plt.savefig(filepath+filename)

# mb_vmat7 = signal_list[6]
# for k in  range(765):
#     sig = mb_vmat7[1][k,:]


#     f,tt,sxx = signal.spectrogram(sig,fs=12800)
#     plt.pcolormesh(tt,f,sxx, shading='gouraud')
#     plt.axis('off')


#     filepath ="C:/Users/ksk/Desktop/hyunmin/A호기/GB_SP_RHS_V.mat/"
#     filename= f'GB_SP_RHS_VMAT{k}.png'
#     plt.savefig(filepath+filename)


# mb_vmat8 = signal_list[7]
# for k in range(607, 765):
#     sig = mb_vmat8[1][k, :]
#
#     f, tt, sxx = signal.spectrogram(sig, fs=12800)
#     plt.pcolormesh(tt, f, sxx, shading='gouraud')
#     plt.axis('off')
#
#     filepath = "/home/sung/바탕화면/mat.image/GE_DE_H.mat/"
#     filename = f'GE_DE_HMAT{k}.png'
#     plt.tight_layout()
#     plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
#     plt.savefig(filepath + filename)
#
# print("adad")

# mb_vmat9=signal_list[8]
# for k in  range(765):
#     sig = mb_vmat9[1][k,:]


#     f,tt,sxx = signal.spectrogram(sig,fs=12800)
#     plt.pcolormesh(tt,f,sxx, shading='gouraud')
#     plt.axis('off')


#     filepath ="C:/Users/ksk/Desktop/hyunmin/A호기/GE_DE_V.mat/"
#     filename= f'GE_DE_VMAT{k}.png'
#     plt.savefig(filepath+filename)


# mb_vmat0=signal_list[9]
# for k in  range(765):
#     sig = mb_vmat0[1][k,:]


#     f,tt,sxx = signal.spectrogram(sig,fs=12800)
#     plt.pcolormesh(tt,f,sxx, shading='gouraud')
#     plt.axis('off')


#     filepath ="C:/Users/ksk/Desktop/hyunmin/A호기/GE_NDE_H.mat/"
#     filename= f'GE_NDE_HMAT{k}.png'
#     plt.savefig(filepath+filename)


# mb_vmat11=signal_list[10]
# for k in  range(765):
#     sig = mb_vmat11[1][k,:]


#     f,tt,sxx = signal.spectrogram(sig,fs=12800)
#     plt.pcolormesh(tt,f,sxx, shading='gouraud')
#     plt.axis('off')


#     filepath ="C:/Users/ksk/Desktop/hyunmin/A호기/GE_NDE_V.mat/"
#     filename= f'GE_NDE_VMAT{k}.png'
#     plt.savefig(filepath+filename)