""" Python module for calculating the volume fraction variance of pixel patterns """
import numpy as np
import scipy.fft
import skimage.morphology
import skimage.io
rn = np.random.default_rng()

img_test = skimage.io.imread(\
    'https://uploads6.wikiart.org/images/ilya-mashkov/landscape-1914.jpg',as_gray=True)

def poisson_point_pattern(num_particles,dims,side=1):
    """
    Generate Poisson point pattern of normalized square particles with
    given side length. Particles are placed randomly with periodic
    boundary conditions.

    Arguments

        n : int
            Number of particles

        dims : tuple of ints_
            Dimensions of the image

        side : int
            Length of side of each particle
        
    Returns

        img : numpy.ndarray
            2D numpy array of floats
    """
    img = np.zeros(dims,dtype=float)
    positions = rn.integers(low=np.zeros([num_particles,2]),high=[[dims[0],dims[1]]]*num_particles)
    for i in range(num_particles):
        img[:side,:side] = img[:side,:side] + 1
        img = np.roll(img,shift=positions[i],axis=(0,1))
    return img


def volume_fraction_variance(image,max_scale,every=1,window_sizes=None):
    """
    Return the volume fraction variance of grayscale image.

    Arguments:

        image : numpy.ndarray
            2D numpy array of floats in [0,1]

        max_scale : float
            Scale of maximum window size expressed as fraction
            of the smaller dimension of the image

        every : int
            Increment for scaling of the window
            Overridden if window_sizes != None
        
        window_sizes : np.array of ints
            Window side lengths for which to calculate the volume
            fraction variance. Overrides the value of 'every' if
            specified.
    """
    dimensions = np.shape(image)
    if not window_sizes:
        window_sizes = np.arange(1,int(max_scale*min(dimensions)),every)
    fft_dims   = [scipy.fft.next_fast_len(d) for d in dimensions]
    fft_image  = scipy.fft.rfft2(image,fft_dims)
    vol_var = np.zeros(len(window_sizes))
    for i,side in enumerate(window_sizes):
        window = skimage.morphology.square(side)
        fft_window = scipy.fft.rfft2(window,fft_dims)
        conv = scipy.fft.irfft2(fft_image*fft_window)/window.sum()
        vol_var[i] = np.var(conv)
    return window_sizes,vol_var

def reproduce_fig2_1707_01524():
    """ Reproduce fig 2 in 1707.01524 (square particles only). """
    dims = (3000,3000)
    num_pixels = np.prod(dims)
    side_particles = np.array([3,30])
    vols_particles = side_particles**2
    phis = [0.02]
    for i,phi in enumerate(phis):
        nums_particles = np.array(phi*num_pixels/vols_particles,dtype=int)
        imgs = [poisson_point_pattern(nums_particles[i],dims,side) for i,side in enumerate(side_particles)]
    return imgs