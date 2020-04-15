# code for fitting spectra, using the models in spectral_model.py
from __future__ import absolute_import, division, print_function # python2 compatibility
import numpy as np
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy import signal
from scipy.stats import norm
from . import spectral_model
from . import utils

#------------------------------------------------------------------------------------------

def fitting_mike(spectrum, spectrum_err, spectrum_blaze,\
                  wavelength, NN_coeffs, wavelength_payne, RV_prefit=False,\
                  blaze_normalized=False,
                  RV_array=np.linspace(-1,1.,6), order_choice=[20]):
    '''
    Fitting MIKE spectrum

    Fitting radial velocity can be very multimodal. The best strategy is to initalize over
    different RVs. When RV_prefit is true, we first fit a single order to estimate
    radial velocity that we will adopt as the initial guess for the global fit.

    RV_array is the range of RV that we will consider
    RV array is in the unit of 100 km/s
    order_choice is the order that we choose to fit when RV_prefit is TRUE

    When blaze_normalized is True, we first normalize spectrum with the blaze

    Returns:
        Best fitted parameter (Teff, logg, Fe/H, Alpha/Fe, polynomial coefficients, vmacro, RV)
    '''

    # normalize wavelength grid
    if RV_prefit:
        spectrum = spectrum[order_choice,:]
        spectrum_err = spectrum_err[order_choice,:]
        spectrum_blaze = spectrum_blaze[order_choice,:]
        wavelength_normalized = utils.whitten_wavelength(wavelength)[order_choice,:]
        wavelength = wavelength[order_choice,:]

    # normalize spectra with the blaze function
    if blaze_normalized:
        spectrum = spectrum/spectrum_blaze
        spectrum_err = spectrum_err/spectrum_blaze

    # number of pixel per order, number of order
    num_pixel = spectrum.shape[1]
    num_order = spectrum.shape[0]

    print(spectrum.shape)
    print(spectrum_err.shape)
    print(wavelength.shape)

#------------------------------------------------------------------------------------------
    # the objective function
    def fit_func(dummy_variable, *labels):

        # make payne models
        full_spec = spectral_model.get_spectrum_from_neural_net(\
                                    scaled_labels = labels[:4],
                                    NN_coeffs = NN_coeffs)

        # broadening kernel
        win = norm.pdf((np.arange(21)-10.)*(wavelength_payne[1]-wavelength_payne[0]),\
                                scale=labels[-2]/3e5*5000)
        win = win/np.sum(win)

        # vbroad -> RV
        full_spec = signal.convolve(full_spec, win, mode='same')
        full_spec = utils.doppler_shift(wavelength_payne, full_spec, labels[-1]*100.)

        # interpolate into the observed wavelength
        f_flux_spec = interpolate.interp1d(wavelength_payne, full_spec)

        # loop over all orders
        spec_predict = np.zeros(num_order*num_pixel)
        for k in range(spectrum.shape[0]):
            scale_poly = wavelength_normalized[k,:]**2*labels[4+3*k] \
                        + wavelength_normalized[k,:]*labels[5+3*k] + labels[6+3*k]
            spec_predict[k*num_pixel:(k+1)*num_pixel] = scale_poly*f_flux_spec(wavelength[k,:])
        return spec_predict

#------------------------------------------------------------------------------------------
    # loop over all possible
    chi_2 = np.inf

    if RV_prefit:
        print('Finding the best initial radial velocity')

    if not(RV_prefit) and blaze_normalized:
        print('First fitting the blaze-normalized spectrum')

    for i in range(RV_array.size):
        print(i, "/", RV_array.size)

        # first four are stellar parametes (Teff, logg, Fe/H, alpha/Fe)
        # then the continuum (quadratic)
        # then vamcro
        # then RV
        p0 = np.zeros(4 + 3*num_order + 1 + 1)
        print(p0.shape)

        # initiate the polynomial with a flat scaling of y=1
        p0[4::3] = 0
        p0[5::3] = 0
        p0[6::3] = 1

        # initializE vmacro
        p0[-2] = 0.5

        # initiate RV
        p0[-1] = RV_array[i]

        # set fitting bound
        bounds = np.zeros((2,p0.size))
        bounds[0,:4] = -0.5
        bounds[1,:4] = 0.5
        bounds[0,4:] = -1000
        bounds[1,4:] = 1000
        bounds[0,-2] = 0.1
        bounds[1,-2] = 10.
        bounds[0,-1] = -200
        bounds[1,-1] = 200

        # run the optimizer
        tol = 5e-4
        popt, pcov = curve_fit(fit_func, xdata=[],\
                               ydata = spectrum.ravel(), sigma = spectrum_err.ravel(),\
                               p0 = p0, bounds=bounds, ftol = tol, xtol = tol, absolute_sigma = True,\
                               method = 'trf')

        # calculate chi^2
        model_spec = fit_func([], *popt)
        chi_2_temp = np.mean((spectrum - model_spec)**2/spectrum_err**2)

        # check if this gives a better fit
        if chi_2_temp < chi_2:
            chi_2 = chi_2_temp
            popt_best = popt

    return popt_best
