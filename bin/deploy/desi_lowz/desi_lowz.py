''' deploy on DESI Year 1 BGS data 
'''
import os, sys
import torch
import numpy as np
from astropy.table import Table as aT

from sedflow import flows as F
from sedflow import galaxy as G

import signal
from contextlib import contextmanager


#################################################################### 
# inputs
#################################################################### 
istart = int(sys.argv[1])
iend   = int(sys.argv[2])

#################################################################### 
# read BGS galaxies 
lowz = aT.read('/tigress/chhahn/sedflow/alexamon_lowz/LOWZ_z_photometry.csv') 
lowz['igal'] = np.arange(len(lowz))

print('Deploying SEDflow on %i to %i out of %i total' % (istart, iend, len(lowz))) 
lowz = lowz[istart:iend]
#################################################################### 
# compile photometry 
#################################################################### 
# 1. compile fluxes 
flux_g = 10**((22.5 - lowz['MAG_G']) * 0.4) # nanomaggies
flux_r = 10**((22.5 - lowz['MAG_R']) * 0.4)
flux_z = 10**((22.5 - lowz['MAG_Z']) * 0.4)
fluxes = np.array([flux_g, flux_r, flux_z, lowz['FLUX_W1'], lowz['FLUX_W2']]).T

# 2. compile uncertainties
sig_g  = np.abs(flux_g) * np.abs(-0.4 * np.log(10) * lowz['MAG_G_ERR'])
sig_r  = np.abs(flux_r) * np.abs(-0.4 * np.log(10) * lowz['MAG_R_ERR'])
sig_z  = np.abs(flux_z) * np.abs(-0.4 * np.log(10) * lowz['MAG_Z_ERR'])
sig_w1 = lowz['FLUX_IVAR_W1']**-0.5
sig_w2 = lowz['FLUX_IVAR_W2']**-0.5

sig_fluxes = np.array([sig_g, sig_r, sig_z, sig_w1, sig_w2]).T

redshift = lowz['Z']
#################################################################### 
# timeout exception for out of distribution 
#################################################################### 
class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

#################################################################### 
# deploy flows 
#################################################################### 
if torch.cuda.is_available(): device = 'cuda'
else: device = 'cpu'

# import flows
gsed = G.ModelB(name='modelb.lowz', device=device)
desiflow = F.DESIflow(gsed=gsed, device=device)

n_sample = 10000
for igal in range(len(lowz)): 
    if not np.all(np.isfinite(sig_fluxes[igal])): continue 
    if redshift[igal] < 0.: continue 
    if redshift[igal] > 1.: continue 
    if fluxes[igal][-2] <= 0: continue # has W1 
    if fluxes[igal][-1] <= 0: continue # has W2 

    fpost = os.path.join('/tigress/chhahn/sedflow/alexamon_lowz/',
                         'desi_lowz.sedflow.modelb.lowz.cdf.grzW1W2.%i.npy' % lowz['igal'][igal])
    if os.path.isfile(fpost): continue 

    try: 
        print('IGAL = %i' % (lowz['igal'][igal]))
        with time_limit(30):     
            posterior_samples = desiflow.run(
                    fluxes[igal],           # flux in nmgy
                    sig_fluxes[igal],       # sigma in nmgy 
                    redshift[igal],         # redshift
                    Nsample=n_sample,       # number of samples 
                    progress_bar=False)

            np.save(fpost, posterior_samples)
    except TimeoutException: 
        print('TARGETID = %i TIMEDOUT' % (lowz['igal'][igal]))
        continue  
