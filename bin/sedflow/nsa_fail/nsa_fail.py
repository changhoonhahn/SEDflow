import os, sys
import numpy as np

from sedflow import obs as Obs
from sedflow import train as Train

from provabgs import infer as Infer
from provabgs import models as Models


####################################################
# input
####################################################
sample  = sys.argv[1]
itrain  = int(sys.argv[2]) 
nhidden = int(sys.argv[3]) 
nblocks = int(sys.argv[4])
niter   = int(sys.argv[5])
i0      = int(sys.argv[6])
i1      = int(sys.argv[7])

####################################################
# compile NSA failures 
####################################################
# u, g, r, i, z, sigma_u, sigma_g, sigma_r, sigma_i, sigma_z, redshift 
y_nsa = Obs.load_nsa_data(test_set=False)

igals = np.load('/scratch/network/chhahn/sedflow/nsa_fail/fail.igals.npy')

# convert to flux 
y_flux = Train.mag2flux(y_nsa[:,:5])
y_ivar = Train.sigma_mag2flux(y_nsa[:,5:10], y_nsa[:,:5])**-2
y_zred = y_nsa[:,-1]

####################################################
# setup inference 
####################################################
# SPS parameter priors 
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

# SPS model 
m_sps = Models.NMF(burst=True, emulator=True)

def run_mcmc(i_obs): 
    # desi MCMC object
    nsa_mcmc = Infer.nsaMCMC(model=m_sps, prior=prior_sps)
    
    fmcmc = os.path.join('/scratch/network/chhahn/sedflow/nsa_fail', 
            'mcmc.nsa.%i.hdf5' % i_obs)

    if not os.path.isfile(fmcmc): 
        print('%s running' % os.path.basename(fmcmc))
    
        if not np.all(np.isfinite(y_flux[i_obs])): 
            print('NaN photometry', y_flux[i_obs])
            return None
        
        if not np.all(np.isfinite(y_ivar[i_obs])): 
            print('NaN ivar', y_ivar[i_obs])
            return None

        # run MCMC
        zeus_chain = nsa_mcmc.run(
                bands='sdss', # u, g, r, i, z
                photo_obs=y_flux[i_obs], 
                photo_ivar_obs=y_ivar[i_obs], 
                zred=y_zred[i_obs],
                vdisp=0.,
                sampler='zeus',
                nwalkers=30,
                burnin=0,
                opt_maxiter=2000,
                niter=niter,
                progress=True,
                writeout=fmcmc)
    else: 
        print('%s already exists' % os.path.basename(fmcmc))
    return None 


for i in range(i0, i1+1): 
    run_mcmc(igals[i])
