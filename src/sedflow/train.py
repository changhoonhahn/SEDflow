'''


module for training SEDflow 


'''
import os
import h5py
import numpy as np


def load_data(train_or_test, version=1, sample='toy', params='props', cuts=None, whiten=False, flux_space=False): 
    ''' training and validation data based on specified split 

    returns
        x_train, y_train, x_valid, y_valid
    '''
    assert np.sum(split) == 1.
    assert train_or_test in ['train', 'test'] 

    x_all, y_all = _load_data(train_or_test, version=version, sample=sample, params=params, flux_space=False) 
    
    if cuts is not None: 
        x_all = x_all[cuts, :]
        y_all = y_all[cuts, :]

    if whiten: 
        x_avg = np.mean(x_all, axis=0)
        x_std = np.std(x_all, axis=0)

        y_avg = np.mean(y_all, axis=0)
        y_std = np.std(y_all, axis=0)

        x_all = (x_all - x_avg) / x_std
        y_all = (y_all - y_avg) / y_std

    return x_all, y_all 


def _load_data(train_or_test, version=1, sample='toy', params='props', flux_space=False): 
    ''' load specific SED data 
    '''
    dat_dir = data_dir()
    
    str_v       = 'v%i' % version 
    str_sample  = {'toy': 'toy_noise', 'flow': 'nsa_flow'}[sample]

    fnpy = lambda s: '%s.%s.%s.%s.npy' % (train_or_test, str_v, s, str_sample)
    
    # read in galaxy properties or theta_SPS 
    thetas = np.load(fnpy(params)) 

    if not flux_space: 
        mags    = np.load(fnpy('mags')) 
        sigs    = np.load(fnpy('sigs')) 
    else: 
        mags    = np.load(fnpy('flux'))
        sigs    = np.load(fnpy('sigs_flux')) 
    zreds   = np.load(fnpy('zred'))

    x = thetas
    y = np.concatenate([mags, sigs, zreds], axis=1) 
    return x, y


# def load_test_mcmc(version='0.1', sample='toy.gold', params='props'): 
#    dat_dir = data_dir() 
#
#    if sample == 'toy.gold': 
#        if params == 'props': # galaxy parameters
#            return np.load(os.path.join(dat_dir, 'mcmc.props.test.toy.gold.npy'))
#        elif params == 'sps_unt': # untransformed SPS parameters
#            return np.load(os.path.join(dat_dir, 'mcmc.thetas_unt.test.toy.gold.npy'))
#        elif params == 'sps': # transformed SPS parameters 
#            return np.load(os.path.join(dat_dir, 'mcmc.thetas_sps.test.toy.gold.npy'))
#    else:
#        raise NotImplementedError


def photometry_bands():
    ''' SDSS u, g, r, i, z  photometric bands for NSA catlaog 
    '''
    from speclite import filters as specFilter
    sdss_u = specFilter.load_filter('sdss2010-u')
    sdss_g = specFilter.load_filter('sdss2010-g')
    sdss_r = specFilter.load_filter('sdss2010-r')
    sdss_i = specFilter.load_filter('sdss2010-i')
    sdss_z = specFilter.load_filter('sdss2010-z')

    return specFilter.FilterSequence([sdss_u, sdss_g, sdss_r, sdss_i, sdss_z])


def SED_to_maggies(wave, sed, filters=None): 
    import astropy.units as U
    _sed = np.atleast_2d(sed) * 1e-17*U.erg/U.s/U.cm**2/U.Angstrom
    _wave = wave * U.Angstrom
    return filters.get_ab_maggies(_sed, wavelength=_wave)


def flux2mag(flux): 
    ''' convert flux in nanomaggies to magnitudes
    '''
    return 22.5 - 2.5 * np.log10(flux)


def mag2flux(mag): 
    ''' convert magnitudes to flux in nanomaggies
    '''
    return 10**(0.4 * (22.5 - mag)) 


def sigma_flux2mag(sigma_flux, flux): 
    ''' convert sigma_flux to sigma_mag
    '''
    return np.abs(-2.5 * (sigma_flux)/flux/np.log(10))


def sigma_mag2flux(sigma_mag, mag): 
    ''' convert sigma_mag to sigma_flux
    '''
    flux = mag2flux(mag)
    return np.abs(flux) * np.abs(-0.4 * np.log(10) * sigma_mag)


def thetas2props(thetas_sps, redshifts): 
    ''' calculate select galaxy properties from SPS parameter and redshift values 
    '''
    m_nmf = SPSmodel_default(emulator=True)

    logm    = thetas_sps[:,0] 
    sfr     = np.array([m_nmf.avgSFR(thetas_sps[i], zred=redshifts[i], dt=1.)
        for i in range(thetas_sps.shape[0])])
    logssfr = np.log10(sfr).flatten() - logm 
    logz_mw = np.log10(np.array([m_nmf.Z_MW(thetas_sps[i], zred=redshifts[i])
        for i in range(thetas_sps.shape[0])])).flatten()
    tau_ism = thetas_sps[:,-2]

    return np.array([logm, logssfr, logz_mw, tau_ism]).T


def flatten_chain(chain): 
    ''' flatten mcmc chain. If chain object is 2D then it assumes it's
    already flattened. 
    '''
    if len(chain.shape) == 2: return chain # already flat 

    s = list(chain.shape[1:])
    s[0] = np.prod(chain.shape[:2]) 
    return chain.reshape(s)


def setup_dataloader(x, y, batch_size=100, shuffle=True, **kwargs):
    import torch
    tensor    = torch.from_numpy(x.astype(np.float32))
    cond      = torch.from_numpy(y.astype(np.float32))
    dataset   = torch.utils.data.TensorDataset(tensor, cond)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


def SPSmodel_default(emulator=True):
    ''' SPS model is PROVABGS SED model without burst 
    '''
    from provabgs import models as Models
    return Models.NMF(burst=True, emulator=emulator)


def prior_default(): 
    ''' prior on SPS parameters
    '''
    from provabgs import infer as Infer
    prior_sps = Infer.load_priors([
        Infer.UniformPrior(7., 12.5, label='sed'),
        Infer.FlatDirichletPrior(4, label='sed'),           # flat dirichilet priors
        Infer.UniformPrior(0., 1., label='sed'),            # burst fraction
        Infer.UniformPrior(1e-2, 13.27, label='sed'),    # tburst
        Infer.LogUniformPrior(4.5e-5, 1.5e-2, label='sed'), # log uniform priors on ZH coeff
        Infer.LogUniformPrior(4.5e-5, 1.5e-2, label='sed'), # log uniform priors on ZH coeff
        Infer.UniformPrior(0., 3., label='sed'),        # uniform priors on dust1
        Infer.UniformPrior(0., 3., label='sed'),        # uniform priors on dust2
        Infer.UniformPrior(-2., 1., label='sed')     # uniform priors on dust_index
    ])
    return prior_sps 


def data_dir(): 
    ''' get main data directory where the files are stored for whichever machine I'm on 
    '''
    dat_dirs = [
            '/global/cscratch1/sd/chahah/arcoiris/sedflow/', # nersc 
            '/tigress/chhahn/arcoiris/sedflow/', # tiger
            '/scratch/network/chhahn/arcoiris/sedflow/', # adroit 
            '/Users/chahah/data/arcoiris/sedflow/' # mbp
            ]
    for _dir in dat_dirs: 
        if os.path.isdir(_dir): return _dir
