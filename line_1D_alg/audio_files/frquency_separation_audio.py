'''

Audio Signal usually occurs in time domain... but to fft is used to see the signal in its frequency domain and operate accordingly.
Frequency domain analysis is easy than time domain analysis.
In general, you need the Fourier transform if you need to look at the frequencies in a signal. If working with a signal in the time domain is difficult, then using the Fourier transform to move it into the frequency domain is worth trying.


steps to be followed her will be

- taking FFt of the audio file
- taking ifft of the each frequency separately (and this frequency seaparaion will be done by forming a function to divide the whole fft into its samples and then applying ifft on each sample to form a channel) making N different channels of the signals that would be contributing to original signal
- now the comparison will starts between the different channels' intensity (amplitude) value by converting each channel into its binary array

'''




import matplotlib.pyplot as plt # for plotting waves
from scipy.fftpack import fft
# SciPy’s fast Fourier transform (FFT) implementation contains more features and is more likely to get bug fixes than NumPy’s implementation.
from scipy.fftpack import ifft
from scipy.io import wavfile  # get the api

# data = original data
# fs = sampling frequency of data .....SAMPLE_RATE determines how many data points the signal uses to represent the sine wave per second. So if the signal had a sample rate of 10 Hz and was a five-second sine wave, then it would have 10 * 5 = 50 data points.
# Ts = sampling interval
# N = = Number of samples
# sec = total audio track/ length of track in secs / duration of audio
# a = first track channel
# b = normalized first channel data on scale of -1 to +1
# c = Fast Fourier Transform of the data
# d = positive values of FFT
# e = Inverse Fourier Transform of the FFT


fs, data = wavfile.read("D:\count.wav") # load the data

# -*- coding: utf-8 -*-
# NOTE: The WAV file must be 16bit-PCM format.
# It must be encoded at 22050Hz.
# Only mono audio files supported at this time.
# Steps to create the file in Audacity:
# 1) Open the WAV file in Audacity.
# 2) If it is stereo, split the Stereo track to mono
# 3) Delete one of the channels or select only one
# 4) Export the file, leave all the file meta data fields blank
# 5) Set the Sample Format to 16-bit PCM
# 6) Export the file, leave all the file meta data fields blank

print ("Frequency sampling", fs)

l_audio = len(data.shape)
print ("Channels", l_audio)

Ts = 1.0/fs # sampling interval in time
print ("Time-step between samples Ts", Ts)

N = data.shape[0]
print ("Complete Samplings N", N)

secs = N / float(fs)
print ("secs", secs)

a = data.T[0] # this is a two channel soundtrack, I get the first track

'''You are FFTing 2 channel data. You should only FFT 1 channel of mono data for the FFT results to make ordinary sense. If you want to process 2 channels of stereo data, you should IFFT(FFT()) each channel separately.
You are using a real fft, which throws away information, and thus makes the fft non-invertible.
If you want to invert, you will need to use an FFT which produces a complex result, and then IFFT this complex frequency domain vector back to the time domain. If you modify the frequency domain vector, make sure it stays conjugate symmetric if you want a strictly real result (minus numerical noise).'''

# you can fft multi-channel data, you just need to use a 2d array

b=[(ele//2**8.)*2-1 for ele in a] # this is 8-bit track, b is now normalized on [-1,1)
c = fft(b) # calculate fourier transform (complex numbers list)

d = len(c)//2  # you only need half of the fft list (real signal symmetry)

# By definition, the FFT is symmetric across data[0] to data[len-1]

e = ifft(c)

plt.subplot(311)
plt.plot(data, "g") # plotting the signal
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(312)
plt.plot(abs(c[:(d-1)]),"r") # plotting the positive fft spectrum
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')

plt.subplot(313)
plt.plot(e, "g") #ploting the inverse fft of the complete fft spectrum
plt.xlabel('Time')
plt.ylabel('Amplitude of regenerated signal')

plt.show()


'''
# Binary array conversion alg
import scipy
import wave
import struct
import numpy
import pylab
from scipy.io import wavfile
rate, data = wavfile.read("D:\count.wav")
filtereddata = numpy.fft.rfft(data, axis=0)
print("original data is")
print (data)
print("fft of data is")
print (filtereddata )
filteredwrite = numpy.fft.irfft(filtereddata, axis=0)
print("ifft of data is")
print (filteredwrite)

'''

