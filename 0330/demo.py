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
parser.add_argument('--file1', type=str, default='./Pic/PEPPER.BMP')
parser.add_argument('--noise_level2_1', type=float, default=1)
parser.add_argument('--noise_level2_2', type=float, default=10)
parser.add_argument('--C1', type=float, default=0.1)
parser.add_argument('--C2', type=float, default=0.01)

# problem 2

# problem 3

# problem 4

opt = parser.parse_args()

if opt.problem == 1:
    img = np.array(Image.open(opt.file1))
    img_fft = np.fft.fft2(img)

    k = gaussian_blur_kernel(21)
    k1 = convert(img.shape, k)
    K = np.fft.fft2(k1)
    
    noise1 = (np.random.rand(img.shape[0], img.shape[1])-0.5)*opt.noise_level2_1
    N1 = np.fft.fft2(noise1)
    noise2 = (np.random.rand(img.shape[0], img.shape[1])-0.5)*opt.noise_level2_2
    N2 = np.fft.fft2(noise2)
    
    img_blurred_fft1 = img_fft*K + N1
    img_blurred1 = np.fft.ifft2(img_blurred_fft1).real.astype(np.uint8)
    img_blurred_fft2 = img_fft*K + N2
    img_blurred2 = np.fft.ifft2(img_blurred_fft2).real.astype(np.uint8)
    
    H1 = equalizer(K, opt.C1)
    H2 = equalizer(K, opt.C2)
    
    img_recon_fft1 = img_blurred_fft1*H1
    img_recon_fft2 = img_blurred_fft1*H2
    img_recon_fft3 = img_blurred_fft2*H1
    img_recon_fft4 = img_blurred_fft2*H2
    
    img_recon1 = np.fft.ifft2(img_recon_fft1).real.astype(np.uint8)
    img_recon2 = np.fft.ifft2(img_recon_fft2).real.astype(np.uint8)
    img_recon3 = np.fft.ifft2(img_recon_fft3).real.astype(np.uint8)
    img_recon4 = np.fft.ifft2(img_recon_fft3).real.astype(np.uint8)
    
    plt.figure(figsize=(3*5, 3*5))
    
    plt.subplot(3, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.subplot(3, 3, 2)
    plt.imshow(img_blurred1, cmap='gray')
    plt.title(fr'blurred image 1 (maximal noise = {0.5*opt.noise_level2_1})')
    plt.subplot(3, 3, 3)
    plt.imshow(img_blurred2, cmap='gray')
    plt.title(fr'blurred image 2 (maximal noise = {0.5*opt.noise_level2_2})')
    
    plt.subplot(3, 3, 4)
    plt.imshow(20*np.log(np.abs(H1)), cmap='gray')
    plt.title(fr'Equalizer 1 ($C = {opt.C1}$)')
    plt.subplot(3, 3, 5)
    plt.imshow(img_recon1, cmap='gray')
    plt.title(fr'Reconstuction 1 (maximal noise = {0.5*opt.noise_level2_1}, $C = {opt.C1}$)')
    plt.subplot(3, 3, 6)
    plt.imshow(img_recon2, cmap='gray')
    plt.title(fr'Reconstuction 2 (maximal noise = {0.5*opt.noise_level2_1}, $C = {opt.C2}$)')
    
    plt.subplot(3, 3, 7)
    plt.imshow(20*np.log(np.abs(H2)), cmap='gray')
    plt.title(fr'Equalizer 2 ($C = {opt.C2}$)')
    plt.subplot(3, 3, 8)
    plt.imshow(img_recon3, cmap='gray')
    plt.title(fr'Reconstuction 3 (maximal noise = {0.5*opt.noise_level2_2}, $C = {opt.C1}$)')
    plt.subplot(3, 3, 9)
    plt.imshow(img_recon4, cmap='gray')
    plt.title(fr'Reconstuction 4 (maximal noise = {0.5*opt.noise_level2_2}, $C = {opt.C2}$)')
    
    if opt.print_out:
        plt.savefig('p1.jpg')
    plt.clf()
elif opt.problem == 2:
    Data = np.array([
        [2, -1, 3], 
        [-1, 3, 5], 
        [0, 2, 4], 
        [4, -2, -1], 
        [1, 0, 4], 
        [-2, 5, 5]
    ])
    Data = Data - Data.mean(axis=0)
    
    U, Sigma, V_tran = np.linalg.svd(Data)
    S = np.zeros((6, 3))
    S[:3, :3] = np.diag(Sigma)
    
    print(f'Using two components')
    S[:, 2] = 0
    
    Reconstuction_result = np.matmul(np.matmul(U, S), V_tran)
    print(Reconstuction_result)
elif opt.problem == 3:
    non_orthog_set = np.empty((5, 13), dtype=np.int32)
    for i in range(5):
        non_orthog_set[i] = np.linspace(0, 12, 13)**i
    re = Gram_Schmidt_Method(non_orthog_set)
    check_table = np.round(np.matmul(re, re.T), 2)
    if opt.print_out:
        print(check_table)