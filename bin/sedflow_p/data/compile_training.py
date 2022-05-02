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
A_spec, h_spec, A_ivar, h_ivar, zred = [], [], [], [], [] 
for ibatch in range(10): 
    f_Aspec = os.path.join(dat_dir, 'train.v0.1.%i.norm_spec.nde_noise.npy' % ibatch) 
    f_hspec = os.path.join(dat_dir, 'train.v0.1.%i.h_spec.nde_noise.npy' % ibatch) 
    f_Aivar = os.path.join(dat_dir, 'train.v0.1.%i.norm_ivar.nde_noise.npy' % ibatch) 
    f_hivar = os.path.join(dat_dir, 'train.v0.1.%i.h_ivar.nde_noise.npy' % ibatch) 

    A_spec.append(np.load(f_Aspec))
    h_spec.append(np.load(f_hspec))
    A_ivar.append(np.load(f_Aivar))
    h_ivar.append(np.load(f_hivar))

    f_zred = os.path.join(dat_dir, 'train.v0.1.%i.redshifts.npy' % ibatch) 
    zred.append(np.load(f_zred))

A_spec = np.concatenate(A_spec, axis=0)
h_spec = np.concatenate(h_spec, axis=0)
A_ivar = np.concatenate(A_ivar, axis=0)
h_ivar = np.concatenate(h_ivar, axis=0)
zred = np.concatenate(zred, axis=0) 

################################################################
# slight clean up  
################################################################
not_nan = (
        ~(np.sum(np.isnan(A_spec)).astype(bool)) &
        ~(np.sum(np.isnan(h_spec), axis=1).astype(bool)) & 
        ~(np.sum(np.isnan(A_ivar)).astype(bool)) & 
        ~(np.sum(np.isnan(h_ivar), axis=1).astype(bool))) 
print('%i of %i have no NaNs' % (np.sum(not_nan), len(not_nan)))

np.save('/scratch/network/chhahn/sedflow/sedflow_p.train.v0.1.theta_sps.npy',
        theta_sps[not_nan]) 
np.save('/scratch/network/chhahn/sedflow/sedflow_p.train.v0.1.theta_unt.npy', 
        theta_unt[not_nan]) 
np.save('/scratch/network/chhahn/sedflow/sedflow_p.train.v0.1.norm_spec.npy', 
        A_spec[not_nan]) 
np.save('/scratch/network/chhahn/sedflow/sedflow_p.train.v0.1.h_spec.npy', 
        h_spec[not_nan]) 
np.save('/scratch/network/chhahn/sedflow/sedflow_p.train.v0.1.norm_ivar.npy', 
        A_ivar[not_nan]) 
np.save('/scratch/network/chhahn/sedflow/sedflow_p.train.v0.1.h_ivar.npy', 
        h_ivar[not_nan]) 
np.save('/scratch/network/chhahn/sedflow/sedflow_p.train.v0.1.zred.npy', 
        zred[not_nan]) 
