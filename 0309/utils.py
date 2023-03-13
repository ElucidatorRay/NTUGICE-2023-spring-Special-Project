import os
import sys
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

RGB2YCBCR = np.array([[0.299, 0.587, 0.114], [-0.169, -0.331, 0.5], [0.5, -0.419, -0.081]])
YCBCR2RGB = np.linalg.inv(RGB2YCBCR)
SOBEL_H = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])/4
SOBEL_V = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])/4
SOBEL_45 = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]])/4
SOBEL_135 = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])/4
LAPLACIAN = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])/8

#############################################################
######################### problem 1 #########################
#############################################################

def difference(img, direction='h'):
    '''
    :param img: {numpy.ndarray} input image matrix (in gray form)
    :param direction: {string} calculation direction of difference
    :return: {numpy.ndarray} edge detection result
    '''
    if direction == 'h':
        re = img - np.roll(img, 1, axis=1)
        re[:, 0] = 0
        return re
    elif direction == 'v':
        re = img - np.roll(img, 1, axis=0)
        re[0, :] = 0
        return re

def _2dconv(img, kernel, padding=0, stride=1):
    '''
    :param img: {numpy.ndarray} input image matrix (in gray form)
    :param kernel: {numpy.ndarray} apllied kernel
    :param padding: {int} must be positive
    :param stride: {int} must be positive
    :return: {numpy.ndarray} convolution result
    '''
    padding = int(padding)
    stride = int(stride)
    
    kernel = np.flipud(np.fliplr(kernel))
    
    H, W = img.shape[0], img.shape[1]
    k_size_h, k_size_w = kernel.shape[0], kernel.shape[1]
    
    re = np.zeros((int(((H - k_size_h + 2*padding)/stride)+1)
                   , int(((W - k_size_w + 2*padding)/stride)+1)))
    
    if padding != 0:
        new_img = np.zeros((img.shape[0] + 2*padding, img.shape[1] + 2*padding))
        new_img[padding:-1*padding, padding:-1*padding] = img
        img = new_img
        
    for i in range(img.shape[0]):
        if i + k_size_h > img.shape[0]:
            break
        if i % stride == 0:
            for j in range(img.shape[1]):
                if j + k_size_w > img.shape[1]:
                    break
                if j % stride == 0:
                    #print(i, i+k_size_h, j, j+k_size_w)
                    re[i//stride, j//stride] = (img[i:i+k_size_h, j:j+k_size_w]*kernel).sum()
    return re

def edge_detect(img, Type):
    '''
    :param img: {numpy.ndarray} input image matrix (in gray form)
    :param Type: {string} using what algorithm or kernel
    :return: {numpy.ndarray} edge detection result
    '''
    global SOBEL_H, SOBEL_V, SOBEL_45, SOBEL_135, LAPLACIAN
    if Type == 'diff_h':
        return difference(img, 'h')
    elif Type == 'diff_v':
        return difference(img, 'v')
    elif Type == 'sobel_h':
        return _2dconv(img, SOBEL_H)
    elif Type == 'sobel_v':
        return _2dconv(img, SOBEL_V)
    elif Type == 'sobel_45':
        return _2dconv(img, SOBEL_45)
    elif Type == 'sobel_135':
        return _2dconv(img, SOBEL_135)
    elif Type == 'laplacian':
        return _2dconv(img, LAPLACIAN)

#############################################################
######################### problem 2 #########################
#############################################################

def rgb2ycbcr(img):
    '''
    :param img: {numpy.ndarray} input image matrix (in RGB form)
    :return: {numpy.ndarray} output image matrix (in YCbCr form)
    '''
    global RGB2YCBCR
    H, W = img.shape[0], img.shape[1]
    
    img = np.transpose(img, (2, 0, 1)).reshape((3, -1))
    img_ycbcr = np.matmul(RGB2YCBCR, img).reshape((3, H, W))
    img_ycbcr = np.transpose(img_ycbcr, (1, 2, 0))
    return img_ycbcr

def ycbcr2rgb(img):
    '''
    :param img: {numpy.ndarray} input image matrix (in RGB form)
    :return: {numpy.ndarray} output image matrix (in RGB form, floored to integer)
    '''
    global YCBCR2RGB
    H, W = img.shape[0], img.shape[1]
    
    img = np.transpose(img, (2, 0, 1)).reshape((3, -1))
    img_rgb = np.matmul(YCBCR2RGB, img).reshape((3, H, W))
    img_rgb = np.transpose(img_rgb, (1, 2, 0))
    img_rgb = img_rgb.astype(np.uint8)
    return img_rgb
    
def luminance_adjust(img, alpha=1, mode='RGB'):
    '''
    :param img: {numpy.ndarray} input image matrix
    :param alpha: {float} luminance adjust factor
    :param mode: {string} input format(RGB, YCbCr, ...)
    :return: {numpy.ndarray} output image
    '''
    if mode == 'RGB':
        img = rgb2ycbcr(img)
        img[:, :, 0] = 255*np.power(img[:, :, 0]/255, alpha)
        img = ycbcr2rgb(img)
        return img
    elif mode == 'YCbCr':
        pass
        img[:, :, 0] = 255*np.power(img[:, :, 0]/255, alpha)
        img = ycbcr2rgb(img)
        return img
    elif mode == 'gray':
        img = 255*np.power(img/255, alpha)
        img = img.astype(np.uint8)
        return img

#############################################################
######################### problem 3 #########################
#############################################################

def error(img1, img2, Type, mode='gray'):
    '''
    compute error of two images
    :param img1: {numpy.ndarray} input image matrix (in RGB form)
    :param img1: {numpy.ndarray} input image matrix (in RGB form)
    :param Type: {string} using what error (maximal, MSE, NMSE, NRMSE, SNR, PSNR, SSIM), using img1 as denominator or molecular
    :return: {float} error
    '''
    H, W = img1.shape[0], img1.shape[1]
    if Type == 'maximal':
        return np.abs(img1 - img2).max()
    elif Type == 'MSE':
        return np.power(img1 - img2, 2).sum() / (img1.shape[0]*img1.shape[1])
    elif Type == 'NMSE':
        return np.power(img1 - img2, 2).sum() / np.power(img1, 2).sum()
    elif Type == 'NRMSE':
        return math.sqrt(np.power(img1 - img2, 2).sum() / np.power(img1, 2).sum())
    elif Type == 'SNR':
        return 10*np.log10( np.power(img1, 2).sum()/np.power(img1 - img2, 2).sum() )
    elif Type == 'PSNR':
        if mode == 'gray':
            return 10*np.log10( H*W*(255**2)/np.power(img1 - img2, 2).sum() )
        elif mode == 'RGB':
            return 10*np.log10( 3*H*W*(255**2)/np.power(img1 - img2, 2).sum() )
    elif Type == 'SSIM':
        mu1 = img1.mean()
        mu2 = img2.mean()
        sig1_s = np.power(img1 - mu1, 2).sum()/H/W
        sig2_s = np.power(img2 - mu2, 2).sum()/H/W
        sig12 = ( (img1-mu1)*(img2-mu2) ).sum()/H/W
        L = 255
        c1, c2 = math.sqrt(1/255), math.sqrt(1/255)
        SSIM = (2*mu1*mu2 + (c1*L)**2)
        SSIM *= (2*sig12 + (c2*L)**2)
        SSIM /= (mu1**2 + mu2**2 + (c1*L)**2)
        SSIM /= (sig1_s + sig2_s + (c2*L)**2)
        return SSIM
