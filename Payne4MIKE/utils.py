# a few low-level functions that are used throughout
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
import os
from scipy import interpolate
from .read_spectrum import read_carpy_fits
from . import spectral_model


#=======================================================================================================================

def vac2air(lamvac):
    """
    http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    Morton 2000
    """
    s2 = (1e4/lamvac)**2
    n = 1 + 0.0000834254 + 0.02406147 / (130 - s2) + 0.00015998 / (38.9 - s2)
    return lamvac/n

def read_in_neural_network():

    '''
    read in the weights and biases parameterizing a particular neural network.
    '''

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'other_data/NN_normalized_spectra_float16.npz')
    tmp = np.load(path)
    w_array_0 = tmp["w_array_0"]
    w_array_1 = tmp["w_array_1"]
    w_array_2 = tmp["w_array_2"]
    b_array_0 = tmp["b_array_0"]
    b_array_1 = tmp["b_array_1"]
    b_array_2 = tmp["b_array_2"]
    x_min = tmp["x_min"]
    x_max = tmp["x_max"]
    wavelength_payne = tmp["wavelength_payne"]
    wavelength_payne = vac2air(wavelength_payne)
    NN_coeffs = (w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max)
    tmp.close()
    return NN_coeffs, wavelength_payne


#--------------------------------------------------------------------------------------------------------------------------

def read_in_example():

    '''
    read in a default spectrum to be fitted.
    '''

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'other_data/2M06062375-0639010_red_multi.fits')
    wavelength, spectrum, spectrum_err = read_carpy_fits(path)
    return wavelength, spectrum, spectrum_err


#--------------------------------------------------------------------------------------------------------------------------

def read_in_blaze_spectrum():

    '''
    read in a default hot star spectrum to determine telluric features and blaze function.
    '''

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'other_data/Hot_Star_HR718.fits')
    wavelength_blaze, spectrum_blaze, spectrum_err_blaze = read_carpy_fits(path)
    return wavelength_blaze, spectrum_blaze, spectrum_err_blaze


#--------------------------------------------------------------------------------------------------------------------------

def doppler_shift(wavelength, flux, dv):

    '''
    dv is in km/s
    positive dv means the object is moving away.
    '''

    c = 2.99792458e5 # km/s
    doppler_factor = np.sqrt((1 - dv/c)/(1 + dv/c))
    new_wavelength = wavelength * doppler_factor
    new_flux = np.interp(new_wavelength, wavelength, flux)
    return new_flux


#--------------------------------------------------------------------------------------------------------------------------

def match_blaze_to_spectrum(wavelength, spectrum, wavelength_blaze, spectrum_blaze):

    '''
    match wavelength of the blaze spectrum to the wavelength of the fitting spectrum
    '''

    for i in range(wavelength.shape[0]):
        if wavelength_blaze[i,0] > wavelength[i,0]:
            wavelength_blaze[i,0] = wavelength[i,0]
        if wavelength_blaze[i,-1] < wavelength[i,-1]:
            wavelength_blaze[i,-1] = wavelength[i,-1]

    spectrum_interp = np.zeros(wavelength.shape)
    for i in range(wavelength.shape[0]):
        f_blaze = interpolate.interp1d(wavelength_blaze[i,:], spectrum_blaze[i,:])
        spectrum_interp[i,:] = f_blaze(wavelength[i,:])

    return spectrum_interp, wavelength


#------------------------------------------------------------------------------------------

def mask_telluric_region(spectrum_err, spectrum_blaze,
                         smooth_length=30, threshold=0.9):

    '''
    mask out the telluric region by setting infinite errors
    '''

    for j in range(spectrum_blaze.shape[0]):
        for i in range(spectrum_blaze[j,:].size-smooth_length):
            if np.min(spectrum_blaze[j,i:i+smooth_length]) \
                    < threshold*np.max(spectrum_blaze[j,i:i+smooth_length]):
                spectrum_err[j,i:i+smooth_length] = 999.
    return spectrum_err

#------------------------------------------------------------------------------------------

def cut_wavelength(wavelength, spectrum, spectrum_err, wavelength_min = 3500, wavelength_max = 10000):

    '''
    remove orders not in wavelength range
    '''

    ii_good = np.sum((wavelength > wavelength_min) & (wavelength < wavelength_max), axis=1) == wavelength.shape[1]
    print("Keeping {}/{} orders between {}-{}".format(ii_good.sum(), len(ii_good), wavelength_min, wavelength_max))
    return wavelength[ii_good,:], spectrum[ii_good,:], spectrum_err[ii_good,:]

#------------------------------------------------------------------------------------------

def mask_wavelength_regions(wavelength, spectrum_err, mask_list):

    '''
    mask out a mask_list by setting infinite errors
    '''

    assert wavelength.shape == spectrum_err.shape
    for wmin, wmax in mask_list:
        assert wmin < wmax
        spectrum_err[(wavelength > wmin) & (wavelength < wmax)] = 999.
    return spectrum_err

#------------------------------------------------------------------------------------------

def scale_spectrum_by_median(spectrum, spectrum_err):

    '''
    dividing spectrum by its median
    '''

    for i in range(spectrum.shape[0]):
        scale_factor = 1./np.median(spectrum[i,:])
        spectrum[i,:] = spectrum[i,:]*scale_factor
        spectrum_err[i,:] = spectrum_err[i,:]*scale_factor
    return spectrum, spectrum_err

#---------------------------------------------------------------------

def whitten_wavelength(wavelength):

    '''
    normalize the wavelength of each order to facilitate the polynomial continuum fit
    '''

    wavelength_normalized = np.zeros(wavelength.shape)
    for k in range(wavelength.shape[0]):
        mean_wave = np.mean(wavelength[k,:])
        wavelength_normalized[k,:] = (wavelength[k,:]-mean_wave)/mean_wave
    return wavelength_normalized

