'''

script to compile all of the training data into some easily useable form 

'''
import os, sys
import numpy as np


dat_dir = '/scratch/network/chhahn/sedflow/training_sed/'
################################################################
# compile thetas
################################################################
fsps = os.path.join(dat_dir, 'train.v0.1.101.thetas_sps.npy')
funt = os.path.join(dat_dir, 'train.v0.1.101.thetas_unt.npy')

theta_sps = np.load(fsps)
theta_unt = np.load(funt)

################################################################
# compile observables 
################################################################

# encoded spectra
fenc = os.path.join(dat_dir, 'train.v0.1.101.encoded.npy') 
enc_spec = np.load(fenc)

# encoded ivar
h_Aivar = np.load('/scratch/network/chhahn/sedflow/spectra/h_Aivar.nde.101.npy') 
enc_ivar = np.concatenate([h_Aivar[:,-1:], h_Aivar[:,:-1]], axis=1) 

# redshift 
fzred = os.path.join(dat_dir, 'train.v0.1.101.redshifts.npy')
zred = np.load(fzred)

################################################################
# slight clean up  
################################################################
not_nan = ~(np.sum(np.isnan(enc_spec), axis=1).astype(bool)) & ~(np.sum(np.isnan(enc_ivar), axis=1).astype(bool)) 
print('%i of %i have no NaNs' % (np.sum(not_nan), len(not_nan)))


np.save('/scratch/network/chhahn/sedflow/sedflow_p.test.v0.1.theta_sps.npy',
        theta_sps[not_nan]) 
np.save('/scratch/network/chhahn/sedflow/sedflow_p.test.v0.1.theta_unt.npy', 
        theta_unt[not_nan]) 
np.save('/scratch/network/chhahn/sedflow/sedflow_p.test.v0.1.encoded.npy', 
        enc_spec[not_nan]) 
np.save('/scratch/network/chhahn/sedflow/sedflow_p.test.v0.1.ivar.encoded.npy', 
        enc_ivar[not_nan]) 
np.save('/scratch/network/chhahn/sedflow/sedflow_p.test.v0.1.zred.npy', 
        zred[not_nan]) 
