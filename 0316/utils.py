import os
import sys
import math
import numpy as np
from PIL import Image
from scipy import signal
import matplotlib.pyplot as plt

#############################################################
######################### problem 1 #########################
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
        
#############################################################
######################### problem 2 #########################
#############################################################

def DFT(x):
    """Function to calculate the DFT
    :param x: {numpy.ndarray} input signal
    """
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    X = np.dot(e, x)
    return X

#############################################################
######################### problem 3 #########################
#############################################################

def h1():
    '''
    :return: {tuple} a two elements tuple, representing time and signal respectively
    '''
    return (np.array([-1, 0, 1]), np.array([1, 0, -1]))

def h2():
    '''
    :return: {tuple} a two elements tuple, representing time and signal respectively
    '''
    hx = np.array([-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
    hy = np.array([1/25, 2/25, 3/25, 4/25, 6/25, 8/25, 0, -8/25, -6/25, -4/25, -3/25, -2/25, -1/25])
    return hx, hy
    
def h3(L, sig):
    '''Function to generate edge detection filter with form h[n] = -C sgn[n] exp(-sig|n|)
    :param L: {int} length of half filter
    :param sig: {float} coef for detecting tiny/large-scale (larger/smaller) edges
    :return: {tuple} a two elements tuple, representing time and signal respectively
    '''
    C = 1/np.exp(-1*sig*np.linspace(1, L, L)).sum()
    hx = np.linspace(-1*L, L, 2*L + 1)
    hy = []
    for elem in hx:
        if elem > 0:
            hy.append( -1*C*1*math.exp(-1*sig*elem) )
        elif elem == 0:
            hy.append(0)
        elif elem < 0:
            hy.append( -1*C*-1*math.exp(sig*elem) )
    return hx, np.array(hy)

def add_noise(signal, an):
    '''Function to adding noise [-0.5*an 0.5*an] to a input signal
    :param signal: {numpy.ndarray} input signal
    :param an: {float} noise amplitude
    :return: {numpy.ndarray} output signal
    '''
    noise = an*np.random.rand(len(signal)) - 0.5
    return signal + noise

#############################################################
######################### problem 4 #########################
#############################################################

def h4(L1):
    '''Function to generate smooth filter with form h[n] = 1 from -L1 to L1 
    :param L1: {int} length of half filter
    :return: {tuple} a two elements tuple, representing time and signal respectively
    '''
    hx = np.linspace(-L1, L1, 2*L1+1)
    hy = np.ones(len(hx))/(2*L1 + 1)
    return hx, hy

def h5():
    ''' 
    :return: {tuple} a two elements tuple, representing time and signal respectively
    '''
    hx = np.arange(9) - 4
    hy = np.array([0.04, 0.08, 0.12, 0.16, 0.2, 0.16, 0.12, 0.08, 0.04])
    return hx, hy

def h6(L, sig):
    '''Function to generate feature extraction filter with form h[n] = C*exp(-sig|n|)
    :param L: {int} length of half filter
    :param sig: {int} scale of feature
    :return: {tuple} a two elements tuple, representing time and signal respectively
    '''
    hx = np.linspace(-L, L, 2*L+1)
    C = 1/(np.exp(-1*sig*np.abs(hx))).sum()
    hy = C*np.exp(-1*sig*np.abs(hx))
    return hx, hy