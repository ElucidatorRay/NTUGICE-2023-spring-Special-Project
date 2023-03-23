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

def gaussian_blur_kernel(n):
    '''Function to generate gaussian blur kernel
    :param n: {int} half length of kernel
    :return: {numpy.ndarray} kernel
    '''
    W = np.repeat(np.arange(n)[:, np.newaxis], n, axis=1) - n//2
    H = np.repeat(np.arange(n)[np.newaxis, :], n, axis=0) - n//2
    re = np.exp(-0.1*(H**2 + W**2))
    s = 1/re.sum()
    return re*s

def convert(shape, kernel):
    '''Function to convert kernel into shape like target image
    :param shape: {tuple} tuple of gray image (H, W)
    :param kernel: {numpy.ndarray} original kernel array
    :return: {numpy.ndarray} converted kernel matrix
    '''
    H, W = shape[0], shape[1]
    L = kernel.shape[0]//2
    
    re = np.zeros(shape)
    re[:L+1, :L+1] = kernel[L:, L:]
    re[:L+1, -L:] = kernel[L:, :L]
    re[-L:, :L+1] = kernel[:L, L:]
    re[-L:, -L:] = kernel[:L, :L]
    return re

def equalizer(K, C):
    '''Function to generate equalizer
    :param K: {numpy.ndarray} spectrum of kernel
    :param C: {float} 
    '''
    K_star = np.empty(K.shape, dtype='complex')
    K_star.real = K.real
    K_star.imag = -1*K.imag
    H = 1/(C/K_star + K)
    return H
    
#############################################################
######################### problem 3 #########################
#############################################################

def Gram_Schmidt_Method(vector_set):
    '''
    :param vector_set: {numpy.ndarray} matrix (vector_index, feature_index) of the non-orthogonal vector set
    :return: {numpy.ndarray} the orthonormal vector set 
    '''
    N = len(vector_set[0])
    phi = ( vector_set[0]/np.sqrt(np.power(vector_set[0], 2).sum()) )[np.newaxis, :]
    for i in range(1, len(vector_set)):
        inner_product = np.sum(np.repeat(vector_set[i][np.newaxis, :], i, axis=0) * phi, axis=1)
        g_a = vector_set[i] - (np.repeat(inner_product[:, np.newaxis], N, axis=1)*phi).sum(axis=0)
        new_phi = g_a/np.inner(g_a, g_a)**0.5
        phi = np.concatenate((phi, new_phi[np.newaxis, :]), axis=0)
    return phi
