"""
Adaptive tone mapping

Adapted from:
https://github.com/IsaacChanghau/OptimizedImageEnhance/blob/master/src/main/java/com/isaac/models/ALTMRetinex.java

Referece: 
http://koasas.kaist.ac.kr/bitstream/10203/172985/1/73275.pdf

"""
import numpy as np
import cv2
from relight.guided_filter import GuidedFilter

def adaptive_tone_map(image, radius=10, eps=0.01, kernel_ratio=0.01, eta=36.0, lamda=10.0):
    """
    Params:
        Image: cv2 image, in RGB
        radius: guided filter radius, defualt 10
        eps: default 0.01
        kernel_ratio: dilation filter kernel, size = kernel_ratio * image_shape
        eta: default 36.0
        lamda: default 10.0
    """
    # global adaptation
    lg, lw = lum_global(image, eta, lamda)
    # guided filter
    hg, dilation = lum_local(lg)
    # l_out
    l_out = lum_out(lg, hg, eta=eta, lamda=lamda)
    # gain
    gain = np.where(lw == 0, l_out, l_out / lw)
    # output image
    channels = cv2.split(image)
    mapped = []
    for ch in cv2.split(image):
        g = ch * gain
        norm = g.max() / 255.0
        mapped.append(g / norm)
        
    # back to RGB
    output = cv2.merge(mapped).astype(np.uint8)
    return output

def lum_global(image):
    """Global adaptation
    Params:
        image: cv2 image in RGB
    """
    # luminance
    lw = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
    lw_max = lw.max()
    # log-average
    lw_avg = np.exp(np.log(lw + 0.001).mean())    
    # global adaptation
    lg = np.log(lw / lw_avg + 1) / np.log(lw_max / lw_avg + 1)
    return lg, lw

def lum_local(lum_global, kernel_ratio=0.01, radius=10, eps=0.01):
    """Local adaptation"""
    kernel_size = max(3.0, kernel_ratio * lum_global.shape[0], kernel_ratio * lum_global.shape[1])
    kernel_size = int(kernel_size)
    kernel = np.ones((kernel_size, kernel_size))
    
    dilation = cv2.dilate(lum_global, kernel, iterations = 1)
    gf = GuidedFilter(dilation, radius=radius, eps=eps)
    hg = gf.filter(lum_global)
    return hg, dilation

def lum_out(lum_global, lum_local, eta=36.0, lamda=10.):
    """Output adaptation"""
    # alpha
    alpha = 1 + eta * (lum_global / lum_global.max())
    # beta
    lg_avg = np.exp(np.log(lum_global + 0.001).mean())
    beta = lamda * lg_avg
    # output
    l_out = alpha * np.log(lum_global / lum_local + beta)
    l_out = cv2.normalize(l_out, 0, 255, cv2.NORM_MINMAX)
    return l_out

