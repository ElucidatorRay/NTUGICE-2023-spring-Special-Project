import os
import sys
import math
import numpy as np
from PIL import Image
import cv2
from scipy import signal
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--problem', type=int)
parser.add_argument('--print_out', type=bool, default=False)

# problem 1
parser.add_argument('--noise_an1', type=float, default=0.1)
parser.add_argument('--filter_length1', type=int, default=5)
parser.add_argument('--spatial_param', type=float, default=0.3)
parser.add_argument('--range_param', type=float, default=5.0)

# problem 2

# problem 3
parser.add_argument('--file3_1', type=str, default='./Pic/PEPPER.BMP')
parser.add_argument('--file3_2', type=str, default='./Pic/lena_256.bmp')
parser.add_argument('--threshold', type=float, default=5)

# problem 4
parser.add_argument('--file4', type=str, default='./test4.png')


opt = parser.parse_args()

if opt.problem == 1:
    Xx = np.linspace(0, 100, 101)
    Yx = np.array([1]*51 + [0]*50)
    Yx_noise = add_noise(Yx, opt.noise_an1)

    re = bilateral_filtering(Yx_noise, opt.filter_length1, opt.spatial_param, opt.range_param)

    plt.plot(Xx, Yx, label='original')
    plt.plot(Xx, Yx_noise, label=fr'original with noise($a_n = {opt.noise_an1}$)')
    plt.plot(range(5, 96), re, label=fr'filtering result($k_1 ={opt.spatial_param}, k_2 = {opt.range_param} $)')
    plt.legend()
    
    if opt.print_out:
        plt.savefig('p1.jpg')
    plt.clf()
elif opt.problem == 2:
    tmp_x = np.linspace(0, 99, 100)
    # 1st pattern
    Y = np.array([0]*100)
    Y[25:75] = 1
    # 2nd pattern
    tmp = (tmp_x*0.04 - 2)
    tmp[:25] = 0
    tmp[75:] = 0
    Y = np.concatenate((Y, tmp), axis=0)
    # matched filter
    tar_x = np.arange(50) - 25
    tar = tmp.copy()
    # 3rd pattern
    tmp = (tmp_x*-1*0.04 + 2)
    tmp[:25] = 0
    tmp[75:] = 0
    Y = np.concatenate((Y, tmp), axis=0)
    # 4th pattern
    tmp = np.zeros(100)
    tmp[25:75] = np.sin(np.arange(50)/50*2*np.pi)
    Y = np.concatenate((Y, tmp), axis=0)
    
    re1 = pattern_recog_norm(Y, tar)
    re2 = pattern_recog_norm_offset(Y, tar)
    rex = np.arange(len(re1)) + tar_x[0]

    plt.figure(figsize=(4*4, 3*4))
    plt.subplot(3, 1, 1)
    plt.plot(np.arange(len(Y)), Y)
    plt.title('Input')
    plt.subplot(3, 1, 2)
    plt.plot(rex, re1)
    plt.title('Result of Correlation in normalization form')
    plt.subplot(3, 1, 3)
    plt.plot(rex, re2)
    plt.title('Result of Correlation in normalization and offset form')
    
    if opt.print_out:
        plt.savefig('p2.jpg')
    plt.clf()
elif opt.problem == 3:
    img1 = np.array(Image.open(opt.file3_1))
    img2 = np.array(Image.open(opt.file3_2))

    mag1, phase1 = mag_angle(img1)
    mag2, _ = mag_angle(img2)

    lpf = low_pass_filter(img1.shape, opt.threshold).astype(np.uint8)
    hpf = high_pass_filter(img1.shape, opt.threshold).astype(np.uint8)

    mag = mag1*lpf + mag2*hpf

    re = reconstruct(mag, phase1)
    
    plt.figure(figsize=(4*4, 2*4))
    plt.subplot(2, 4, 1)
    plt.imshow(img1, cmap='gray')
    plt.title('Image 1')
    plt.subplot(2, 4, 2)
    plt.imshow(20*np.log(mag1), cmap='gray')
    plt.title('Magnitude of image 1')
    plt.subplot(2, 4, 3)
    plt.imshow(lpf, cmap='gray')
    plt.title('Low pass filter')
    plt.subplot(2, 4, 5)
    plt.imshow(img2, cmap='gray')
    plt.title('image 2')
    plt.subplot(2, 4, 6)
    plt.imshow(20*np.log(mag2), cmap='gray')
    plt.title('Magnitude of image 2')
    plt.subplot(2, 4, 7)
    plt.imshow(hpf, cmap='gray')
    plt.title('High pass filter')
    plt.subplot(2, 4, 8)
    plt.imshow(re, cmap='gray')
    plt.title('Combined result')
    
    if opt.print_out:
        plt.savefig('p3.jpg')
    plt.clf()
elif opt.problem == 4:
    img = cv2.imread(opt.file4, 0)
    img[img < 128] = 0
    img[img >= 128] = 1
    
    ero3 = erosion(erosion(erosion(img)))
    dil3 = dilation(dilation(dilation(img)))
    opening = dilation(dilation(dilation(erosion(erosion(erosion(img))))))
    closing = erosion(erosion(erosion(dilation(dilation(dilation(img))))))
    
    plt.figure(figsize=(5*4, 1*4))
    plt.subplot(1, 5, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.subplot(1, 5, 2)
    plt.imshow(ero3, cmap='gray')
    plt.title('Erosion x3')
    plt.subplot(1, 5, 3)
    plt.imshow(dil3, cmap='gray')
    plt.title('Dilation x3')
    plt.subplot(1, 5, 4)
    plt.imshow(opening, cmap='gray')
    plt.title('Opening')
    plt.subplot(1, 5, 5)
    plt.imshow(closing, cmap='gray')
    plt.title('Closing')
    if opt.print_out:
        plt.savefig('p4.jpg')
    plt.clf()