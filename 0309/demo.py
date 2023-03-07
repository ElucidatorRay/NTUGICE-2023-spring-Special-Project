import os
import sys
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--problem', type=int)
parser.add_argument('--print_out', type=bool, default=False)

# problem 1
parser.add_argument('--file1', type=str, default='./Pic/PEPPER.bmp')
parser.add_argument('--C', type=float, default=1.0)

# problem 2
parser.add_argument('--file2', type=str, default='./Pic/PEPPER.bmp')
parser.add_argument('--mode', type=str, default='gray')
parser.add_argument('--alpha_l', type=float, default=0.4)
parser.add_argument('--alpha_d', type=float, default=2.0)

# problem 3
parser.add_argument('--file3_1', type=str, default='./Pic/lena512.bmp')
parser.add_argument('--file3_2', type=str, default='./Pic/lena512c.bmp')

opt = parser.parse_args()

if opt.problem == 1:
    i = np.array(Image.open(opt.file1))
    dh = edge_detect(i, 'diff_h')
    dv = edge_detect(i, 'diff_v')
    sh = edge_detect(i, 'sobel_h')
    sv = edge_detect(i, 'sobel_v')
    s45 = edge_detect(i, 'sobel_45')
    s135 = edge_detect(i, 'sobel_135')
    l = edge_detect(i, 'laplacian')

    C = 1

    plt.figure(figsize=(4*4, 2*4))
    plt.subplot(2, 4, 1)
    plt.imshow(C*np.abs(dh), cmap='gray')
    plt.title('Difference (horizontal)', fontsize=12)

    plt.subplot(2, 4, 2)
    plt.imshow(C*np.abs(dv), cmap='gray')
    plt.title('Difference (vertical)', fontsize=12)

    plt.subplot(2, 4, 3)
    plt.imshow(C*np.abs(sh), cmap='gray')
    plt.title('Sobel operator (horizontal)', fontsize=12)

    plt.subplot(2, 4, 4)
    plt.imshow(C*np.abs(sv), cmap='gray')
    plt.title('Sobel operator (vertical)', fontsize=12)

    plt.subplot(2, 4, 5)
    plt.imshow(C*np.abs(s45), cmap='gray')
    plt.title('Sobel operator (45 degrees)', fontsize=12)

    plt.subplot(2, 4, 6)
    plt.imshow(C*np.abs(s135), cmap='gray')
    plt.title('Sobel operator (135 degrees)', fontsize=12)

    plt.subplot(2, 4, 7)
    plt.imshow(C*np.abs(l), cmap='gray')
    plt.title('laplacian operator', fontsize=12)
    
    plt.subplot(2, 4, 8)
    plt.imshow(i, cmap='gray')
    plt.title('original')
    
    if opt.print_out:
        plt.savefig('p1.jpg')
    plt.clf()
elif opt.problem == 2:
    i = Image.open(opt.file2)
    l = Image.fromarray( luminance_adjust(np.array(i), alpha=opt.alpha_l, mode=opt.mode) )
    d = Image.fromarray( luminance_adjust(np.array(i), alpha=opt.alpha_d, mode=opt.mode) )
    plt.figure(figsize=(3*4, 1*4))
    plt.subplot(1, 3, 1)
    plt.imshow(i, cmap='gray')
    plt.title(fr'original')

    plt.subplot(1, 3, 2)
    plt.imshow(d, cmap='gray')
    plt.title(fr'darken ($\alpha = ${opt.alpha_d})')
    
    plt.subplot(1, 3, 3)
    plt.imshow(l, cmap='gray')
    plt.title(fr'lighten ($\alpha = ${opt.alpha_l})')

    if opt.print_out:
        plt.savefig('p2.jpg')
    plt.clf()
elif opt.problem == 3:
    img1 = np.array( Image.open(opt.file3_1) )
    img2 = np.array( Image.open(opt.file3_2) )
    
    img1 = img1.astype(np.int64)
    img2 = img2.astype(np.int64)
    
    #print( error(img1, img2, Type='maximal', mode='RGB') )
    #print( error(img1, img2, Type='MSE', mode='RGB') )
    #print( error(img1, img2, Type='NMSE', mode='RGB') )
    print( error(img1, img2, Type='NRMSE', mode='RGB') )
    #print( error(img1, img2, Type='SNR', mode='RGB') )
    print( error(img1, img2, Type='PSNR', mode='RGB') )
    #print( error(img1, img2, Type='SSIM', mode='RGB') )