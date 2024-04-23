'''

script to forward model photometry


'''
import os,sys
import psutil
import numpy as np
import multiprocessing as mp 

from speclite import filters as specFilter

from astropy import units as aU

from sedflow import util as U

name    = sys.argv[1]
seed    = int(sys.argv[2])
band    = sys.argv[3] 

n_cpu   = psutil.cpu_count(logical=False)
dat_dir = os.path.join(U.data_dir(), 'seds', name)

#############################################
# set bandpasses 
#############################################
if band == 'ugrizJ': 
    # set up photometric filters u, g, r, i, z, J
    sdss_u = specFilter.load_filter('sdss2010-u')
    sdss_g = specFilter.load_filter('sdss2010-g')
    sdss_r = specFilter.load_filter('sdss2010-r')
    sdss_i = specFilter.load_filter('sdss2010-i')
    sdss_z = specFilter.load_filter('sdss2010-z')
    hsc_y = specFilter.load_filter('hsc2017-y')

    bandpasses = specFilter.FilterSequence([sdss_u, sdss_g, sdss_r, sdss_i, sdss_z, hsc_y])
elif band == 'grzW1W2W3W4': 
    # set up photometric filters DECam DR1 g, r, z with WISE W1, W2, W3, W4
    # this is ultimately what we want to use with DESI photometry 
    if name == 'modela': 
        raise ValueError("Model A does not include dust emission so it cannot reliably model W3 and W4 bands")
    decam_g = specFilter.load_filter('decamDR1-g')
    decam_r = specFilter.load_filter('decamDR1-r')
    decam_z = specFilter.load_filter('decamDR1-z')
    wise_w1 = specFilter.load_filter('wise2010-W1')
    wise_w2 = specFilter.load_filter('wise2010-W2')
    wise_w3 = specFilter.load_filter('wise2010-W3')
    wise_w4 = specFilter.load_filter('wise2010-W4')

    bandpasses = specFilter.FilterSequence([decam_g, decam_r, decam_z,
                                            wise_w1, wise_w2, wise_w3, wise_w4])
elif band == 'grzW1W2': 
    # set up photometric filters DECam DR1 g, r, z with WISE W1, W2
    # this is probably the best we can do with Model A
    decam_g = specFilter.load_filter('decamDR1-g')
    decam_r = specFilter.load_filter('decamDR1-r')
    decam_z = specFilter.load_filter('decamDR1-z')
    wise_w1 = specFilter.load_filter('wise2010-W1')
    wise_w2 = specFilter.load_filter('wise2010-W2')
    bandpasses = specFilter.FilterSequence([decam_g, decam_r, decam_z,
                                            wise_w1, wise_w2])
else: 
    raise ValueError("specify one of the bands") 

#############################################
# read wavelengths and SEDs
#############################################
wave = np.load(os.path.join(dat_dir, 'train_sed.%s.%i.waves.npz' % (name, seed)))['arr_0']
seds = np.load(os.path.join(dat_dir, 'train_sed.%s.%i.seds.npz' % (name, seed)))['arr_0']

#############################################
# calcluate magnitudes 
#############################################
def fm_maggies(ii): 
    # convolve redshifted SEds with bnadpasses
    _maggies = bandpasses.get_ab_maggies(seds[ii] * 1e-17*aU.erg/aU.s/aU.cm**2/aU.Angstrom, 
                                         wavelength=wave[ii] * aU.Angstrom)
    return np.array(list(_maggies[0])) 


with mp.Pool(n_cpu) as p: 
    wfps = p.map(fm_maggies, np.arange(seds.shape[0]))

maggies = np.array(wfps) * 1e9 # to nMgy
#mags = 22.5 - 2.5 * np.log10(maggies) # convert maggies to magnitudes 

# save to file 
np.save(os.path.join(dat_dir, 'train_sed.%s.%i.nmgy_%s.npy' % (name, seed, band)), maggies)
