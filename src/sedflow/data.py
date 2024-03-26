'''

module for SEDflow training/test data 


'''
import os
import numpy as np

from . import util as U


def load_modela(train_or_test, bands='grzW1W2', infer_redshift=False): 
    ''' load training/testing data for SEDflow Model A

    Returns
    -------
    x, y for p(x|y) 
    '''
    if bands not in ['grzW1W2', 'ugrizJ']: raise ValueError
    if train_or_test == 'train': 
        seeds = np.arange(10) 
    elif train_or_test == 'test': 
        seeds = [999]
    else: 
        raise ValueError
    
    dat_dir = os.path.join(U.data_dir(), 'seds') 

    theta, zred, nmgy, sigs = [], [], [], []
    for seed in seeds: 
        _theta = np.load(os.path.join(dat_dir, 'modela', 'train_sed.modela.%i.thetas_unt.npz' % seed))['arr_0']
        _zred = np.load(os.path.join(dat_dir, 'modela', 'train_sed.modela.%i.redshifts.npz' % seed))['arr_0']
        _nmgy = np.load(os.path.join(dat_dir, 'modela', 'train_sed.modela.%i.nmgy_noisy_%s.npy' % (seed, bands)))
        _sigs = np.load(os.path.join(dat_dir, 'modela', 'train_sed.modela.%i.sig_nmgy_%s.npy' % (seed, bands)))
        
        theta.append(_theta)
        zred.append(_zred)
        nmgy.append(_nmgy)
        sigs.append(_sigs)

    theta = np.concatenate(theta)
    # put metallicities in log10 space 
    theta[:,6] = np.log10(theta[:,6])
    theta[:,7] = np.log10(theta[:,7])

    zred = np.concatenate(zred) 
    nmgy = np.concatenate(nmgy)
    sigs = np.concatenate(sigs)


    if not infer_redshift: 
        x = theta
        y = np.concatenate([nmgy, sigs, zred[:,None]], axis=1) 
    else: 
        x = np.concatenate([theta, zred[:,None]], axis=1)
        y = np.concatenate([nmgy, sigs], axis=1) 
    return x, y
