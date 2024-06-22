''' deploy on DESI Year 1 BGS data 
'''
import os, sys
import torch
import numpy as np
from astropy.table import Table as aT

from sedflow import flows as F

import signal
from contextlib import contextmanager


#################################################################### 
# inputs
#################################################################### 
istart = int(sys.argv[1])
iend   = int(sys.argv[2])

#################################################################### 
# read BGS galaxies 
bgs = aT.read('/global/cfs/projectdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.2/BGS_ANY_full.dat.fits')

# select only BGS objects with good spectra with successful redshifts
is_good = (
    (bgs['COADD_FIBERSTATUS'] == 0) &
    (bgs['SPECTYPE'] == 'GALAXY') &
    (bgs['Z_not4clus'] > 0.) &
    (bgs['ZWARN'] == 0) &
    (bgs['DELTACHI2'] > 40.) &
    (bgs['ZERR'] < 0.0005 * (1 + bgs['Z_not4clus'])) 
    )

#bgs = bgs[is_good][1000000*iblock:1000000*(iblock+1)]
print('Deploying SEDflow on %i to %i out of %i total' % (istart, iend, np.sum(is_good))) 
bgs = bgs[is_good][istart:iend]

#################################################################### 
# compile photometry 
#################################################################### 
# 1. correct for extinction
flux_g  = bgs['FLUX_G']/bgs['MW_TRANSMISSION_G']
flux_r  = bgs['FLUX_R']/bgs['MW_TRANSMISSION_R']
flux_z  = bgs['FLUX_Z']/bgs['MW_TRANSMISSION_Z']
flux_w1 = bgs['FLUX_W1']/bgs['MW_TRANSMISSION_W1']
flux_w2 = bgs['FLUX_W2']/bgs['MW_TRANSMISSION_W2']

fluxes = np.array([flux_g, flux_r, flux_z, flux_w1, flux_w2]).T

# 2. compile other measurements 
sig_g  = bgs['FLUX_IVAR_G']**-0.5
sig_r  = bgs['FLUX_IVAR_R']**-0.5
sig_z  = bgs['FLUX_IVAR_Z']**-0.5
sig_w1 = bgs['FLUX_IVAR_W1']**-0.5
sig_w2 = bgs['FLUX_IVAR_W2']**-0.5

sig_fluxes = np.array([sig_g, sig_r, sig_z, sig_w1, sig_w2]).T

redshift = bgs['Z_not4clus'] 

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
desiflow = F.DESIflow(name='modelb.lowz.grzW1W2', device=device)

n_sample = 10000
for igal in range(len(bgs)): 
    if not np.all(np.isfinite(sig_fluxes[igal])): continue 
    if redshift[igal] < 0.: continue 
    if redshift[igal] > 1.: continue 

    fpost = os.path.join('/global/cfs/projectdirs/desi/users/chahah/sedflow/desiy1',
                         'desiy1.bgs.sedflow.modelb.lowz.cdf.grzW1W2.%i.npy' % bgs['TARGETID'][igal])
    if os.path.isfile(fpost): continue 

    try: 
        print('TARGETID = %i' % (bgs['TARGETID'][igal]))
        with time_limit(1):     
            posterior_samples = desiflow.run(
                    fluxes[igal],           # flux in nmgy
                    sig_fluxes[igal],       # sigma in nmgy 
                    redshift[igal],         # redshift
                    Nsample=n_sample,       # number of samples 
                    progress_bar=False)

            np.save(fpost, posterior_samples)
    except TimeoutException: 
        print('TARGETID = %i TIMEDOUT' % (bgs['TARGETID'][igal]))
        continue  

