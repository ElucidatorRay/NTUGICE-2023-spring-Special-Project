import os
import sys
import math
import numpy as np
from PIL import Image
from scipy import signal
import matplotlib.pyplot as plt

EROSION_DILATION_KERNEL = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

#############################################################
######################### problem 1 #########################
#############################################################

def add_noise(signal, an):
    '''Function to adding noise [-0.5*an 0.5*an] to a input signal
    :param signal: {numpy.ndarray} input signal
    :param an: {float} noise amplitude
    :return: {numpy.ndarray} output signal
    '''
    noise = an*(np.random.rand(len(signal)) - 0.5)
    return signal + noise

def bilateral_filtering(signal, L_half, k1, k2):
    '''Function of doing the non-padding 1-D bilateral filtering
    :param signal: {numpy.ndarray} input signal
    :param L_half: {int} half length of filter
    :param k1: {float} parameter of spatial kernel
    :param k2: {float} parameter of range kernel
    :return: {numpy.ndarray} output signal
    '''
    # init Gs (spatial kernel)
    Gs = np.exp(-1*k1*(np.arange(2*L_half+1) - L_half)**2)
    Gs = np.repeat(Gs[np.newaxis, :], len(signal)-2*L_half, axis=0)
    
    # init Gr (range kernel)
    diff = np.lib.stride_tricks.sliding_window_view(signal, 11) - np.repeat(signal[5:-5][:, np.newaxis], 11, axis=1)
    Gr = np.exp(-1*k2*(diff)**2)
    
    re = np.sum(np.lib.stride_tricks.sliding_window_view(signal, 11)*Gs*Gr, axis=1) / np.sum(Gs*Gr, axis=1)
    return re
    
#############################################################
######################### problem 2 #########################
#############################################################

def pattern_recog_norm(x, pattern):
    '''
    :param x: {numpy.ndarray} input signal
    :param pattern: {numpy.ndarray} desired pattern
    '''
    pattern = pattern[::-1]
    sig_conv_tar = signal.convolve(x, pattern)
    
    x_padded = np.pad(x, len(pattern)-1, mode='constant')
    sig_x = np.sum(np.lib.stride_tricks.sliding_window_view(x_padded, len(pattern))**2, axis=1)**0.5
    sig_h = ((pattern**2).sum())**0.5
    
    sig_x[sig_x == 0] = 1
    return sig_conv_tar/sig_x/sig_h
    
def pattern_recog_norm_offset(x, pattern):
    '''
    :param x: {numpy.ndarray} input signal
    :param pattern: {numpy.ndarray} desired pattern
    '''
    x_padded = np.pad(x, len(pattern)-1, mode='constant')
    x_seg = np.lib.stride_tricks.sliding_window_view(x_padded, len(pattern))
    x0_seg = np.repeat(  ( np.sum(x_seg, axis=1)/len(pattern) )[:, np.newaxis], len(pattern), axis=1  )
    
    h0 = pattern.sum()/len(pattern)
    up = np.sum((x_seg - x0_seg)*(np.repeat(pattern[np.newaxis, :], x_seg.shape[0], axis=0) - h0), axis=1)
    
    sig_x = np.sqrt(np.sum((x_seg - x0_seg)**2, axis=1))
    sig_h = np.sqrt( ((pattern - h0)**2).sum() )
    down = sig_x*sig_h
    down[down == 0] = 1
    
    return up/down

#############################################################
######################### problem 3 #########################
#############################################################

def low_pass_filter(img_size, threshold):
    '''Function to generate low pass filter mask for image
    :param img_size: {tuple} tuple of image size 
    :param threshold: {int} filtering threshold
    :return: {numpy.ndarray} matrix of filter with size of img_size
    '''
    P = np.repeat(np.arange(img_size[0])[:, np.newaxis], img_size[1], axis=1)
    Q = np.repeat(np.arange(img_size[1])[np.newaxis, :], img_size[0], axis=0)
    
    re = np.logical_or((P+Q) <= threshold, (P+(img_size[1]-1-Q)) <= threshold)
    re = np.logical_or(re, ((img_size[0]-1-P) + Q) <= threshold)
    re = np.logical_or(re, ((img_size[0]-1-P) + (img_size[1]-1-Q)) <= threshold)
    
    return re

def high_pass_filter(img_size, threshold):
    '''Function to generate high pass filter mask for image
    :param img_size: {tuple} tuple of image size 
    :param threshold: {int} filtering threshold
    :return: {numpy.ndarray} matrix of filter with size of img_size
    '''
    return np.logical_not(low_pass_filter(img_size, threshold))

def mag_angle(img):
    '''
    :param img: {numpy.ndarray} input image array
    :return: {tuple(numpy.ndarray, numpy.ndarray)} 1st term is magnitude, 2nd term is phase
    '''
    img_fft = np.fft.fft2(img)
    return np.abs(img_fft), np.angle(img_fft)

def reconstruct(mag, angle):
    '''
    :param mag: {numpy.ndarray} magnitude for reconstructing image
    :param angle: {numpy.ndarray} phase for reconstructing image
    '''
    re = np.empty(mag.shape, dtype='complex')
    re.real = mag*np.cos(angle)
    re.imag = mag*np.sin(angle)
    
    re = np.fft.ifft2(re).real.astype(np.uint8)
    return re

#############################################################
######################### problem 4 #########################
#############################################################

def erosion(img):
    '''Funtion for doing erosion on binary image 
    :param img: {numpy.ndarray} input binary image
    :return: {numpy.ndarray} output binary image
    '''
    img = np.pad(img, (1, 1), mode='constant')
    
    global EROSION_DILATION_KERNEL
    img_patch = np.lib.stride_tricks.sliding_window_view(img, (3, 3))
    kernel_patch = np.repeat(EROSION_DILATION_KERNEL[np.newaxis, :, :], img_patch.shape[1], axis=0)
    kernel_patch = np.repeat(kernel_patch[np.newaxis, :, :, :], img_patch.shape[0], axis=0)
    
    re = np.einsum('...ij->...', img_patch*kernel_patch)
    re[re < 5] = 0
    re[re >= 5] = 1
    return re

def dilation(img):
    '''Funtion for doing dilation on binary image 
    :param img: {numpy.ndarray} input binary image
    :return: {numpy.ndarray} output binary image
    '''
    img = np.pad(img, (1, 1), mode='constant')
    
    global EROSION_DILATION_KERNEL
    img_patch = np.lib.stride_tricks.sliding_window_view(img, (3, 3))
    kernel_patch = np.repeat(EROSION_DILATION_KERNEL[np.newaxis, :, :], img_patch.shape[1], axis=0)
    kernel_patch = np.repeat(kernel_patch[np.newaxis, :, :, :], img_patch.shape[0], axis=0)
    
    re = np.einsum('...ij->...', img_patch*kernel_patch)
    re[re > 0] = 1
    return re