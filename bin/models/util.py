'''


some utility funcitions used in training SEDflow models.


'''
import os
import glob
import numpy as np 

from scipy.special import erf, erfinv

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from sedflow import flows as F


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
    
    dat_dir = os.path.join(data_dir(), 'seds') 

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
    nmgy = np.log10(np.concatenate(nmgy)) # convert to log-nanomaggies to reduce dynamic range 
    sigs = np.concatenate(sigs)


    if not infer_redshift: 
        x = theta
        y = np.concatenate([nmgy, sigs, zred[:,None]], axis=1) 
    else: 
        x = np.concatenate([theta, zred[:,None]], axis=1)
        y = np.concatenate([nmgy, sigs], axis=1) 
    return x, y


def load_modelb(train_or_test, bands='grzW1W2', infer_redshift=False): 
    ''' load training/testing data for SEDflow Model B

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
    
    dat_dir = os.path.join(data_dir(), 'seds') 

    theta, zred, nmgy, sigs = [], [], [], []
    for seed in seeds: 
        _theta = np.load(os.path.join(dat_dir, 'modelb', 'train_sed.modelb.%i.thetas_unt.npz' % seed))['arr_0']
        _zred = np.load(os.path.join(dat_dir, 'modelb', 'train_sed.modelb.%i.redshifts.npz' % seed))['arr_0']
        _nmgy = np.load(os.path.join(dat_dir, 'modelb', 'train_sed.modelb.%i.nmgy_noisy_%s.npy' % (seed, bands)))
        _sigs = np.load(os.path.join(dat_dir, 'modelb', 'train_sed.modelb.%i.sig_nmgy_%s.npy' % (seed, bands)))
        
        theta.append(_theta)
        zred.append(_zred)
        nmgy.append(_nmgy)
        sigs.append(_sigs)

    theta = np.concatenate(theta)
    # put metallicities in log10 space 
    theta[:,6] = np.log10(theta[:,6])
    theta[:,7] = np.log10(theta[:,7])

    zred = np.concatenate(zred) 
    nmgy = np.log10(np.concatenate(nmgy)) # convert to log-nanomaggies to reduce dynamic range 
    sigs = np.concatenate(sigs)


    if not infer_redshift: 
        x = theta
        y = np.concatenate([nmgy, sigs, zred[:,None]], axis=1) 
    else: 
        x = np.concatenate([theta, zred[:,None]], axis=1)
        y = np.concatenate([nmgy, sigs], axis=1) 
    return x, y


def load_modelc(train_or_test, bands='grzW1W2', infer_redshift=False): 
    ''' load training/testing data for SEDflow Model B

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
    
    dat_dir = os.path.join(data_dir(), 'seds') 

    theta, zred, nmgy, sigs = [], [], [], []
    for seed in seeds: 
        _theta = np.load(os.path.join(dat_dir, 'modelc', 'train_sed.modelc.%i.thetas_unt.npz' % seed))['arr_0']
        _zred = np.load(os.path.join(dat_dir, 'modelc', 'train_sed.modelc.%i.redshifts.npz' % seed))['arr_0']
        _nmgy = np.load(os.path.join(dat_dir, 'modelc', 'train_sed.modelc.%i.nmgy_noisy_%s.npy' % (seed, bands)))
        _sigs = np.load(os.path.join(dat_dir, 'modelc', 'train_sed.modelc.%i.sig_nmgy_%s.npy' % (seed, bands)))
        
        theta.append(_theta)
        zred.append(_zred)
        nmgy.append(_nmgy)
        sigs.append(_sigs)

    theta = np.concatenate(theta)
    # put metallicities in log10 space 
    theta[:,6] = np.log10(theta[:,6])
    theta[:,7] = np.log10(theta[:,7])

    zred = np.concatenate(zred) 
    nmgy = np.log10(np.concatenate(nmgy)) # convert to log-nanomaggies to reduce dynamic range 
    sigs = np.concatenate(sigs)


    if not infer_redshift: 
        x = theta
        y = np.concatenate([nmgy, sigs, zred[:,None]], axis=1) 
    else: 
        x = np.concatenate([theta, zred[:,None]], axis=1)
        y = np.concatenate([nmgy, sigs], axis=1) 
    return x, y


def data_dir(): 
    ''' get main data directory where the files are stored for whichever machine I'm on 
    '''
    dat_dirs = [
            '/pscratch/sd/c/chahah/sedflow/',  # perlmutter
            '/tigress/chhahn/sedflow/', # tiger
            '/scratch/network/chhahn/sedflow/', # adroit 
            '/Users/chahah/data/sedflow/' # mbp
            ]
    for _dir in dat_dirs: 
        if os.path.isdir(_dir): return _dir


def read_best_qphi(study_name, n_ensemble=5, device='cpu', name='modela'): 
    dat_dir = os.path.join(data_dir(), 'qphi', name) 

    fevents = glob.glob(os.path.join(dat_dir, '%s/*/events*' % study_name))

    events, best_valid = [], []
    for fevent in fevents: 
        ea = EventAccumulator(fevent)
        ea.Reload()

        try: 
            best_valid.append(ea.Scalars('best_validation_log_prob')[0].value)
            events.append(fevent)
        except: 
            pass #print(fevent)

    best_valid = np.array(best_valid)
    print('%i models trained' % np.max([int(os.path.dirname(event).split('.')[-1]) 
        for event in events]))
    
    i_models = [int(os.path.dirname(events[i]).split('.')[-1]) for i 
            in np.argsort(best_valid)[-n_ensemble:][::-1]]
    print(i_models) 
    
    qphis = []
    for i_model in i_models: 
        fqphi = os.path.join(dat_dir, '%s/%s.%i.pt' % (study_name, study_name, i_model))
        qphi = torch.load(fqphi, map_location=device)
        qphis.append(qphi)

    return qphis


def read_best_fqphi(study_name, n_ensemble=5, name='modela'): 
    dat_dir = os.path.join(data_dir(), 'qphi', name) 

    fevents = glob.glob(os.path.join(dat_dir, '%s/*/events*' % study_name))

    events, best_valid = [], []
    for fevent in fevents: 
        ea = EventAccumulator(fevent)
        ea.Reload()

        try: 
            best_valid.append(ea.Scalars('best_validation_log_prob')[0].value)
            events.append(fevent)
        except: 
            pass #print(fevent)

    best_valid = np.array(best_valid)
    print('%i models trained' % np.max([int(os.path.dirname(event).split('.')[-1]) 
        for event in events]))
    
    i_models = [int(os.path.dirname(events[i]).split('.')[-1]) for i 
            in np.argsort(best_valid)[-n_ensemble:][::-1]]
    print(i_models) 
    
    fqphis = []
    for i_model in i_models: 
        fqphis.append(os.path.join(dat_dir, '%s/%s.%i.pt' % (study_name, study_name, i_model)))
    
    return fqphis


def read_EnsembleFlow(study_name, n_ensemble=5, device='cpu', name='modela'): 
    ''' read in ensemble of best flows
    '''
    fqphis = read_best_fqphi(study_name, n_ensemble=n_ensemble, name=name)

    flos = []     
    for fqphi in fqphis: 
        _flo = F.Flow(device=device) 
        _flo.load_flow(fqphi)

        flos.append(_flo)
    return flos 

