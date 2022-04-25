'''

script to compile all of the training data into some easily useable form 

'''
import os, sys
import numpy as np


dat_dir = '/scratch/network/chhahn/sedflow/training_sed/'
################################################################
# compile thetas
################################################################
theta_sps, theta_unt = [], [] 
for ibatch in range(10): 
    fsps = os.path.join(dat_dir, 'train.v0.1.%i.thetas_sps.npy' % ibatch)
    funt = os.path.join(dat_dir, 'train.v0.1.%i.thetas_unt.npy' % ibatch)

    theta_sps.append(np.load(fsps))
    theta_unt.append(np.load(funt))

theta_sps = np.concatenate(theta_sps, axis=0) 
theta_unt = np.concatenate(theta_unt, axis=0) 

################################################################
# compile observables 
################################################################

# encoded spectra
enc_spec = [] 
for ibatch in range(10): 
    fenc = os.path.join(dat_dir, 'train.v0.1.%i.encoded.npy' % ibatch) 
    enc_spec.append(np.load(fenc))

enc_spec = np.concatenate(enc_spec, axis=0) 


# encoded ivar
h_Aivar = [] 
for ibatch in range(10): 
    # read in (h, Aivar) values 
    for isplit in range(10): 
        if ibatch == 0: 
            _hA = np.load('/scratch/network/chhahn/sedflow/spectra/h_Aivar.nde.%iof10.npy' % (isplit+1))
        else: 
            _hA = np.load('/scratch/network/chhahn/sedflow/spectra/h_Aivar.nde.%i.%iof10.npy' % (ibatch, isplit+1))
        if isplit == 0: 
            _h_Aivar = np.zeros((_hA.shape[0]*10, 6))
        _h_Aivar[isplit::10] = _hA
    h_Aivar.append(_h_Aivar)

h_Aivar = np.concatenate(h_Aivar, axis=0) 
enc_ivar = np.concatenate([h_Aivar[:,-1:], h_Aivar[:,:-1]], axis=1) 


# redshift 
zred = []
for ibatch in range(10): 
    fzred = os.path.join(dat_dir, 'train.v0.1.%i.redshifts.npy' % ibatch)
    zred.append(np.load(fzred))
zred = np.concatenate(zred, axis=0) 


################################################################
# slight clean up  
################################################################
not_nan = ~(np.sum(np.isnan(enc_spec), axis=1).astype(bool)) & ~(np.sum(np.isnan(enc_ivar), axis=1).astype(bool)) 
print('%i of %i have no NaNs' % (np.sum(not_nan), len(not_nan)))


np.save('/scratch/network/chhahn/sedflow/sedflow_p.train.v0.1.theta_sps.npy',
        theta_sps[not_nan]) 
np.save('/scratch/network/chhahn/sedflow/sedflow_p.train.v0.1.theta_unt.npy', 
        theta_unt[not_nan]) 
np.save('/scratch/network/chhahn/sedflow/sedflow_p.train.v0.1.encoded.npy', 
        enc_spec[not_nan]) 
np.save('/scratch/network/chhahn/sedflow/sedflow_p.train.v0.1.ivar.encoded.npy', 
        enc_ivar[not_nan]) 
np.save('/scratch/network/chhahn/sedflow/sedflow_p.train.v0.1.zred.npy', 
        zred[not_nan]) 
