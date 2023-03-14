import os
import sys
import math
import numpy as np
from PIL import Image
from scipy import signal
from scipy.io.wavfile import read
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--problem', type=int)
parser.add_argument('--print_out', type=bool, default=False)

# problem 1
parser.add_argument('--file1_1', type=str, default='./Pic/lena512.bmp')
parser.add_argument('--file1_2', type=str, default='./Pic/lena512c.bmp')

# problem 2
parser.add_argument('--file2', type=str, default='./M1F1-uint8WE-AFsp.wav')
parser.add_argument('--length', type=int, default=1024)

# problem 3
parser.add_argument('--L3', type=int, default=20)
parser.add_argument('--sig3_1', type=float, default=0.5)
parser.add_argument('--sig3_2', type=float, default=0.1)
parser.add_argument('--noise_an3_1', type=float, default=0.5)
parser.add_argument('--noise_an3_2', type=float, default=0.7)

# problem 4
parser.add_argument('--L4', type=int, default=20)
parser.add_argument('--sig4_1', type=float, default=0.1)
parser.add_argument('--sig4_2', type=float, default=1)
parser.add_argument('--noise_an4_1', type=float, default=5)
parser.add_argument('--noise_an4_2', type=float, default=10)

opt = parser.parse_args()

if opt.problem == 1:
    img1 = np.array( Image.open(opt.file1_1) )
    img2 = np.array( Image.open(opt.file1_2) )
    
    img1 = img1.astype(np.int64)
    img2 = img2.astype(np.int64)
    
    print( error(img1, img2, Type='SSIM', mode='RGB') )
elif opt.problem == 2:
    rate, data = read(opt.file2)
    data = data.T[0][:opt.length]
    # data_fft = fft(data)
    data_dft = DFT(data)
    mag = np.abs(data_dft)
    plt.plot(np.arange(len(data))[1:], mag[1:])
    plt.ylabel('Magnitude', fontsize=12)
    if opt.print_out:
        plt.savefig('p2.jpg')
    plt.clf()
elif opt.problem == 3:
    Xx = np.linspace(-30, 100, 131)
    Xy = np.array([0]*20 + [1]*31 + [0]*29 + [1]*31 + [0]*20)
    
    h1x, h1y = h3(opt.L3, opt.sig3_1)
    h2x, h2y = h3(opt.L3, opt.sig3_2)
    
    Xy_noise1 = add_noise(Xy, opt.noise_an3_1)
    Xy_noise2 = add_noise(Xy, opt.noise_an3_2)
    
    Y1y = signal.convolve(Xy_noise1, h1y)
    Y2y = signal.convolve(Xy_noise2, h1y)
    Y3y = signal.convolve(Xy_noise1, h2y)
    Y4y = signal.convolve(Xy_noise2, h2y)
    
    Yx = np.linspace(Xx[0] + h1x[0], Xx[0] + h1x[0] + len(Y1y)-1, len(Y1y))
    
    plt.figure(figsize=(4*4, 2*4))
    
    plt.subplot(2, 4, 1)
    plt.scatter(Xx, Xy, s=12)
    plt.title('Original')
    
    plt.subplot(2, 4, 2)
    plt.scatter(Xx, Xy_noise1, s=12)
    plt.title(fr'Add noise with an = {opt.noise_an3_1}')
    
    plt.subplot(2, 4, 3)
    plt.scatter(Xx, Xy_noise2, s=12)
    plt.title(fr'Add noise with an = {opt.noise_an3_2}')
    
    plt.subplot(2, 4, 5)
    plt.scatter(Yx, Y1y, s=12)
    plt.title(fr'result (an = {opt.noise_an3_1}, $\sigma$ = {opt.sig3_1})')
    
    plt.subplot(2, 4, 6)
    plt.scatter(Yx, Y2y, s=12)
    plt.title(fr'result (an = {opt.noise_an3_2}, $\sigma$ = {opt.sig3_1})')
    
    plt.subplot(2, 4, 7)
    plt.scatter(Yx, Y3y, s=12)
    plt.title(fr'result (an = {opt.noise_an3_1}, $\sigma$ = {opt.sig3_2})')
    
    plt.subplot(2, 4, 8)
    plt.scatter(Yx, Y4y, s=12)
    plt.title(fr'result (an = {opt.noise_an3_2}, $\sigma$ = {opt.sig3_2})')
    
    if opt.print_out:
        plt.savefig('p3.jpg')
    plt.clf()
elif opt.problem == 4:
    Xx = np.linspace(-50, 100, 151)
    Xy = 0.1*Xx

    Xy_noise1 = add_noise(Xy, opt.noise_an4_1)
    Xy_noise2 = add_noise(Xy, opt.noise_an4_2)

    h1x, h1y = h6(opt.L4, opt.sig4_1)
    h2x, h2y = h6(opt.L4, opt.sig4_2)

    Y1y = signal.convolve(Xy_noise1, h1y)
    Y2y = signal.convolve(Xy_noise2, h1y)
    Y3y = signal.convolve(Xy_noise1, h2y)
    Y4y = signal.convolve(Xy_noise2, h2y)

    Yx = np.linspace(Xx[0] + h1x[0], Xx[0] + h1x[0] + len(Y1y)-1, len(Y1y))

    plt.figure(figsize=(4*4, 2*4))

    plt.subplot(2, 4, 1)
    plt.scatter(Xx, Xy, s=12)
    plt.title('Original')

    plt.subplot(2, 4, 2)
    plt.scatter(Xx, Xy_noise1, s=12)
    plt.title(fr'Add noise with an = {opt.noise_an4_1}')

    plt.subplot(2, 4, 3)
    plt.scatter(Xx, Xy_noise2, s=12)
    plt.title(fr'Add noise with an = {opt.noise_an4_2}')

    plt.subplot(2, 4, 5)
    plt.scatter(Yx, Y1y, s=12)
    plt.title(fr'result (an = {opt.noise_an4_1}, $\sigma$ = {opt.sig4_1})')

    plt.subplot(2, 4, 6)
    plt.scatter(Yx, Y2y, s=12)
    plt.title(fr'result (an = {opt.noise_an4_2}, $\sigma$ = {opt.sig4_1})')

    plt.subplot(2, 4, 7)
    plt.scatter(Yx, Y3y, s=12)
    plt.title(fr'result (an = {opt.noise_an4_1}, $\sigma$ = {opt.sig4_2})')

    plt.subplot(2, 4, 8)
    plt.scatter(Yx, Y4y, s=12)
    plt.title(fr'result (an = {opt.noise_an4_2}, $\sigma$ = {opt.sig4_2})')
    
    if opt.print_out:
        plt.savefig('p4.jpg')
    plt.clf()