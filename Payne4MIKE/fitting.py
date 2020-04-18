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

def fit_global(spectrum, spectrum_err, spectrum_blaze, wavelength,
                NN_coeffs, wavelength_payne,\
                RV_array=np.linspace(-1,1.,6), order_choice=[20],\
                polynomial_order=6, bounds_set=None):

    '''
    Fitting MIKE spectrum
    Fitting stellar labels, polynomial continuum, vbroad, and radial velocity simultaneously

    spectrum and spectrum_err are the spectrum to be fitted and its uncertainties
    spectrum_blaze is a hot star spectrum used as a reference spectrum
    wavelength is the wavelength of the pixels
    they are all 2D array with [number of spectral order, number of pixels]

    NN_coeffs are the neural network emulator coefficients
    we adopt Kurucz models in this study
    wavelength_payne is the output wavelength of the emulator

    RV_array is the radial velocity initialization that we will run the fit
    order_choice is the specific order that we will use to pre-determine the radial velocitiy
    the radial velocity is then used as the initialization for the global fit
    MIKE spectrum typically has about ~35 orders in the red

    polynomial_order is the final order of polynomial that we will assume for the continuum of individual orders
    A 6th order polynomial does a decent job
    '''

    # first we fit for a specific order while looping over all RV initalization
    # the spectrum is pre-normalized with the blaze function
    # we assume a quadratic polynomial for the residual continuum
    popt_best, model_spec_best, chi_square = fitting_mike(spectrum, spectrum_err, spectrum_blaze,\
                                                          wavelength, NN_coeffs, wavelength_payne,\
                                                          p0_initial=None, RV_prefit=True, blaze_normalized=True,\
                                                          RV_array=RV_array, polynomial_order=2, bounds_set=bounds_set)

    # we then fit for all the orders
    # we adopt the RV from the previous fit as the sole initialization
    # the spectrum is still pre-normalized by the blaze function
    RV_array = np.array([popt_best[-1]])
    popt_best, model_spec_best, chi_square = fitting_mike(spectrum, spectrum_err, spectrum_blaze,\
                                                          wavelength, NN_coeffs, wavelength_payne,\
                                                          p0_initial=None, RV_prefit=False, blaze_normalized=True,\
                                                          RV_array=RV_array, polynomial_order=2, bounds_set=bounds_set)

    # using this fit, we can subtract the raw spectrum with the best fit model of the normalized spectrum
    # with which we can then estimate the continuum for the raw specturm
    poly_initial = fit_continuum(spectrum, spectrum_err, wavelength, popt_best,\
                                         model_spec_best, polynomial_order=polynomial_order, previous_polynomial_order=2)

    # using all these as intialization, we are ready to do the final fit
    RV_array = np.array([popt_best[-1]])
    p0_initial = np.concatenate([popt_best[:4], poly_initial.ravel(), popt_best[-2:]])
    popt_best, model_spec_best, chi_square = fitting_mike(spectrum, spectrum_err, spectrum_blaze,\
                                                          wavelength, NN_coeffs, wavelength_payne,\
                                                          p0_initial=p0_initial, bounds_set=bounds_set,\
                                                          RV_prefit=False, blaze_normalized=False,\
                                                          RV_array=RV_array, polynomial_order=polynomial_order)
    return popt_best, model_spec_best, chi_square

#------------------------------------------------------------------------------------------

def fit_continuum(spectrum, spectrum_err, wavelength, previous_poly_fit, previous_model_spec,\
                  polynomial_order=6, previous_polynomial_order=2):

    '''
    Fit the continuum while fixing other stellar labels

    The end results will be used as initial condition in the global fit (continuum + stellar labels)
    '''

    print('Pre Fit: Finding the best continuum initialization')

    # normalize wavelength grid
    wavelength_normalized = utils.whitten_wavelength(wavelength)*100.

    # number of polynomial coefficients
    coeff_poly = polynomial_order + 1
    pre_coeff_poly = previous_polynomial_order + 1

    # initiate results array for the polynomial coefficients
    fit_poly = np.zeros((wavelength_normalized.shape[0],coeff_poly))

    # loop over all order and fit for the polynomial (weighted by the error)
    for k in range(wavelength_normalized.shape[0]):
        pre_poly = 0
        for m in range(pre_coeff_poly):
            pre_poly += (wavelength_normalized[k,:]**m)*previous_poly_fit[4+m+pre_coeff_poly*k]
        substract_factor =  (previous_model_spec[k,:]/pre_poly) ## subtract away the previous fit
        fit_poly[k,:] = np.polyfit(wavelength_normalized[k,:], spectrum[k,:]/substract_factor,\
                                   polynomial_order, w=1./(spectrum_err[k,:]/substract_factor))[::-1]

    return fit_poly


#------------------------------------------------------------------------------------------

def fitting_mike(spectrum, spectrum_err, spectrum_blaze,\
                 wavelength, NN_coeffs, wavelength_payne, p0_initial=None, bounds_set=None,\
                 RV_prefit=False, blaze_normalized=False, RV_array=np.linspace(-1,1.,6),\
                 polynomial_order=2, order_choice=[20]):

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
    wavelength_normalized = utils.whitten_wavelength(wavelength)*100.

    # number of polynomial coefficients
    coeff_poly = polynomial_order + 1

    # specify a order for the (pre-) RV fit
    if RV_prefit:
        spectrum = spectrum[order_choice,:]
        spectrum_err = spectrum_err[order_choice,:]
        spectrum_blaze = spectrum_blaze[order_choice,:]
        wavelength_normalized = wavelength_normalized[order_choice,:]
        wavelength = wavelength[order_choice,:]

    # normalize spectra with the blaze function
    if blaze_normalized:
        spectrum = spectrum/spectrum_blaze
        spectrum_err = spectrum_err/spectrum_blaze

    # number of pixel per order, number of order
    num_pixel = spectrum.shape[1]
    num_order = spectrum.shape[0]

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
            scale_poly = 0
            for m in range(coeff_poly):
                scale_poly += (wavelength_normalized[k,:]**m)*labels[4+coeff_poly*k+m]
            spec_predict[k*num_pixel:(k+1)*num_pixel] = scale_poly*f_flux_spec(wavelength[k,:])
        return spec_predict

#------------------------------------------------------------------------------------------
    # loop over all possible
    chi_2 = np.inf

    if RV_prefit:
        print('Pre Fit: Finding the best radial velocity initialization')

    if not(RV_prefit) and blaze_normalized:
        print('Pre Fit: Fitting the blaze-normalized spectrum')

    if not(RV_prefit) and not(blaze_normalized):
        print('Final Fit: Fitting the whole spectrum with all parameters simultaneously')

    for i in range(RV_array.size):
        print(i+1, "/", RV_array.size)

        # initialize the parameters (Teff, logg, Fe/H, alpha/Fe, polynomial continuum, vbroad, RV)
        if p0_initial is None:
            p0 = np.zeros(4 + coeff_poly*num_order + 1 + 1)
            p0[4::coeff_poly] = 1
            p0[5::coeff_poly] = 0
            p0[6::coeff_poly] = 0
            p0[-2] = 0.5
            p0[-1] = RV_array[i]
        else:
            p0 = p0_initial

        # set fitting bound
        bounds = np.zeros((2,p0.size))
        bounds[0,4:] = -1000 # polynomial coefficients
        bounds[1,4:] = 1000
        if bounds_set is None:
            bounds[0,:4] = -0.5 # teff, logg, feh, alphafe
            bounds[1,:4] = 0.5
            bounds[0,-2] = 0.1 # vbroad
            bounds[1,-2] = 10.
            bounds[0,-1] = -2. # RV [100 km/s]
            bounds[1,-1] = 2.
        else:
            bounds[:,:4] = bounds_set[:,:4]
            bounds[:,-2:] = bounds_set[:,-2:]

        if (not(bounds_set is None)) and (p0_initial is None):
            p0[:4] = np.mean(bounds_set[:,:4], axis=0)

        # run the optimizer
        tol = 5e-4
        popt, pcov = curve_fit(fit_func, xdata=[],\
                               ydata = spectrum.ravel(), sigma = spectrum_err.ravel(),\
                               p0 = p0, bounds=bounds, ftol = tol, xtol = tol, absolute_sigma = True,\
                               method = 'trf')

        # calculate chi^2
        model_spec = fit_func([], *popt)
        chi_2_temp = np.mean((spectrum.ravel() - model_spec)**2/spectrum_err.ravel()**2)

        # check if this gives a better fit
        if chi_2_temp < chi_2:
            chi_2 = chi_2_temp
            model_spec_best = model_spec
            popt_best = popt

    return popt_best, model_spec_best.reshape(num_order,num_pixel), chi_2
