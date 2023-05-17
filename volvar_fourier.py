""" Python module for calculating the volume fraction variance of pixel patterns """
import numpy as np
import scipy.fft
import skimage.morphology
import skimage.io
import matplotlib.pyplot as pl
rn = np.random.default_rng()

img_test = skimage.io.imread(\
    'https://uploads6.wikiart.org/images/ilya-mashkov/landscape-1914.jpg',as_gray=True)

def poisson_point_pattern(n,dims,side=1,normalize=True):
    """
    Generate Poisson point pattern of square particles with
    given side length.

    Arguments

        n : int
            Number of particles

        dims : tuple of ints
            Dimensions of the image

        side : int
            Length of side of each particle
        
        normalize : bool
            Whether to normalize the image

    Returns

        img : numpy.ndarray
            2D numpy array of floats
    """
    img = np.zeros(dims,dtype=float)
    positions = rn.integers(low=np.zeros([2,n]),high=[[dims[0]]*n,[dims[1]]*n])
    for i in range(n):
        tmp = np.zeros(dims)
        tmp[positions[0,i],positions[1,i]] = 1
        img = img + skimage.morphology.dilation(tmp,skimage.morphology.square(side))
    if normalize:
        img = img/img.sum()
    return img


def volume_fraction_variance(image,scale,every):
    """
    Return the volume fraction variance of grayscale image.

    Arguments:

        image : numpy.ndarray
            2D numpy array of floats in [0,1]

        scale : float
            Scale of maximum window size expressed as fraction
            of the smaller dimension of the image

        every : int
            Increment for scaling of the window
    """
    dimensions = np.shape(image)
    mean = np.mean(image)
    window_sizes = np.arange(1,int(scale*min(dimensions)),every)
    fft_dims   = [scipy.fft.next_fast_len(d) for d in dimensions]
    fft_image  = scipy.fft.rfft2(image,fft_dims)
    vol_var = np.zeros(len(window_sizes))
    for i,side in enumerate(window_sizes):
        window = skimage.morphology.square(side)
        fft_window = scipy.fft.rfft2(window,fft_dims)
        conv = scipy.fft.irfft2(fft_image*fft_window)/window.sum()
        vol_var[i] = np.var(conv)
    return window_sizes,vol_var
