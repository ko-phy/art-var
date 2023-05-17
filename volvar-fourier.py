import numpy as np
import scipy.fft
import skimage.morphology

def volume_fraction_variance(image,scale,every):
    """
    Calculates the volume fraction variance in grayscale image.

    Arguments:

        image : numpy.ndarray
            2D numpy array of floats in [0,1].

        scale : float
            Scale of maximum window size expressed as fraction
            of the smaller dimension of the image.

        every : int
            Increment for scaling of the window.
    """
    dimensions = np.shape(image)
    mean = np.mean(image)
    window_sizes = np.arange(1,int(scale*min(dimensions)),every)
    fft_dims   = [scipy.fft.next_fast_len(d) for d in dimensions]
    fft_image  = scipy.fft.rfft2(image,fft_dims,norm='ortho')
    vol_var = np.zeros(len(window_sizes))
    for i,side in enumerate(window_sizes):
        window = skimage.morphology.square(side)
        fft_window = scipy.fft.rfft2(window,fft_dims,norm='ortho')
        conv = scipy.fft.irfft2( fft_image*fft_window,norm='ortho' )
        vol_var[i] = np.var(conv)
    return window_sizes,vol_var
