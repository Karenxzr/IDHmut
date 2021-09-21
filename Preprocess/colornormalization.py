# Color Normalization and augmentation'
import numpy as np

def rgb2hsv(rgb):
    """ convert RGB to HSV color space
    :param rgb: np.ndarray
    :return: np.ndarray
    """

    rgb = rgb.astype('float')
    maxv = np.amax(rgb, axis=2)
    maxc = np.argmax(rgb, axis=2)
    minv = np.amin(rgb, axis=2)
    minc = np.argmin(rgb, axis=2)

    hsv = np.zeros(rgb.shape, dtype='float')
    hsv[maxc == minc, 0] = np.zeros(hsv[maxc == minc, 0].shape)
    hsv[maxc == 0, 0] = (((rgb[..., 1] - rgb[..., 2]) * 60.0 / (maxv - minv + np.spacing(1))) % 360.0)[maxc == 0]
    hsv[maxc == 1, 0] = (((rgb[..., 2] - rgb[..., 0]) * 60.0 / (maxv - minv + np.spacing(1))) + 120.0)[maxc == 1]
    hsv[maxc == 2, 0] = (((rgb[..., 0] - rgb[..., 1]) * 60.0 / (maxv - minv + np.spacing(1))) + 240.0)[maxc == 2]
    hsv[maxv == 0, 1] = np.zeros(hsv[maxv == 0, 1].shape)
    hsv[maxv != 0, 1] = (1 - minv / (maxv + np.spacing(1)))[maxv != 0]
    hsv[..., 2] = maxv

    return hsv

def hsv2rgb(hsv):
    """ convert HSV to RGB color space
    :param hsv: np.ndarray
    :return: np.ndarray
    """

    hi = np.floor(hsv[..., 0] / 60.0) % 6
    hi = hi.astype('uint8')
    v = hsv[..., 2].astype('float')
    f = (hsv[..., 0] / 60.0) - np.floor(hsv[..., 0] / 60.0)
    p = v * (1.0 - hsv[..., 1])
    q = v * (1.0 - (f * hsv[..., 1]))
    t = v * (1.0 - ((1.0 - f) * hsv[..., 1]))

    rgb = np.zeros(hsv.shape)
    rgb[hi == 0, :] = np.dstack((v, t, p))[hi == 0, :]
    rgb[hi == 1, :] = np.dstack((q, v, p))[hi == 1, :]
    rgb[hi == 2, :] = np.dstack((p, v, t))[hi == 2, :]
    rgb[hi == 3, :] = np.dstack((p, q, v))[hi == 3, :]
    rgb[hi == 4, :] = np.dstack((t, p, v))[hi == 4, :]
    rgb[hi == 5, :] = np.dstack((v, p, q))[hi == 5, :]

    return rgb



def color_augmentation(rgb,sigma=0.05,alpha0=None,beta0=None,alpha1=None,beta1=None,alpha2=None,beta2=None):
    inputimg_hsv=rgb2hsv(rgb)

    if alpha0==None:
        for i in range(3):
            alpha = np.random.uniform(1-sigma,1+sigma)
            beta = np.random.uniform(-sigma,sigma)
            inputimg_hsv[:,:,i] = inputimg_hsv[:,:,i]*alpha+beta
    else:
        inputimg_hsv[:,:,0] = inputimg_hsv[:,:,0]*alpha0+beta0
        inputimg_hsv[:,:,1] = inputimg_hsv[:,:,1]*alpha1+beta1
        inputimg_hsv[:,:,2] = inputimg_hsv[:,:,2]*alpha2+beta2

    outputimg = hsv2rgb(inputimg_hsv)
    return outputimg
