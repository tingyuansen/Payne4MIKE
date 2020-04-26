# import packages
import numpy as np
import time
import os
from astropy.io import fits
from multiprocessing import Pool

# set number of threads per CPU
os.environ['OMP_NUM_THREADS']='{:d}'.format(1)
import mkl
mkl.set_num_threads(1)

# import The Payne (https://github.com/tingyuansen/Payne4MIKE)
from Payne4MIKE import utils
from Payne4MIKE import fitting_lamost


#====================================================================================
# assuming Kurucz models
NN_coeffs, wavelength_payne = utils.read_in_neural_network()
w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max = NN_coeffs

# restore catalog
hdulist = fits.open('../Lamost_DR5_x_APOGEE_DR16.fits')
lmjd = hdulist[1].data['lmjd'][:3*10**4]
planid = hdulist[1].data['planid'][:3*10**4]
spid = hdulist[1].data['spid'][:3*10**4]
fiberid = hdulist[1].data['fiberid'][:3*10**4]

lamost_rv = hdulist[1].data['lamost_rv'][:3*10**4]

#-------------------------------------------------------------------------------------
# perfort the fit in batch
def fit_spectrum(i):

    # load spectrum
    filename = ("spec-%(d)5d-%(s)s_sp%(p)02d-%(f)03d.fits.gz" %\
                {'d': lmjd[i], 's': planid[i].strip(),\
                 'p': spid[i], 'f': fiberid[i]})
    temp = np.load("../lamost_DR5_x_apogee_DR16_input/" + filename + ".npz")
    wavelength = temp["wavelength"]
    spectrum = temp["spectrum"]
    spectrum_err = temp["spectrum_err"]
    spectrum_blaze = temp["spectrum_blaze"]

    # the range of RV that we will search (in the unit of 100 km/s)
    # expand/refine the range of RV if the fit is stuck in a local minimum
    #RV_array = np.linspace(-4,2.,31)
    RV_array = np.array([lamost_rv[i]])/100.

    # fit spectrum
    popt_best, model_spec_best, chi_square = fitting_lamost.fit_global(spectrum, spectrum_err,\
                                                            spectrum_blaze, wavelength,\
                                                            NN_coeffs, wavelength_payne, RV_array=RV_array,\
                                                            polynomial_order=6, order_choice=[1])

    # save results
    np.savez("../payne4lamost_results/" + filename,\
             popt_best=popt_best,\
             model_spec_best=model_spec_best,\
             chi_square=chi_square)

#-------------------------------------------------------------------------------------
# fit spectra in batch
num_CPU = 64
pool = Pool(num_CPU)
start_time = time.time()
pool.map(fit_spectrum,range(lmjd.size));
print('Run time', time.time()-start_time)
