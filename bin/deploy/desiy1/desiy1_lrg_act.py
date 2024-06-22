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
lrgs = aT.read('/global/cfs/projectdirs/desi/users/chahah/sedflow/desiy1/lrg_act/overlapping_catalog_legacy_info.txt', 
               format='ascii')
lrgs['igal'] = np.arange(len(lrgs))

print('Deploying SEDflow on %i to %i out of %i total' % (istart, iend, len(lrgs))) 
lrgs = lrgs[istart:iend]
#################################################################### 
# compile photometry 
#################################################################### 
# 1. correct for extinction
flux_g = lrgs['col5'] / lrgs['col10']
flux_r = lrgs['col6'] / lrgs['col11']
flux_z = lrgs['col7'] / lrgs['col12']
flux_w1 = lrgs['col8'] / lrgs['col13']
flux_w2 = lrgs['col9'] / lrgs['col14']

fluxes = np.array([flux_g, flux_r, flux_z, flux_w1, flux_w2]).T

# 2. compile other measurements
sig_g  = lrgs['col15']**-0.5
sig_r  = lrgs['col16']**-0.5
sig_z  = lrgs['col17']**-0.5
sig_w1 = lrgs['col18']**-0.5
sig_w2 = lrgs['col19']**-0.5

sig_fluxes = np.array([sig_g, sig_r, sig_z, sig_w1, sig_w2]).T

redshift = lrgs['col4']
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
for igal in range(len(lrgs)): 
    if not np.all(np.isfinite(sig_fluxes[igal])): continue 
    if redshift[igal] < 0.: continue 
    if redshift[igal] > 1.: continue 

    fpost = os.path.join('/global/cfs/projectdirs/desi/users/chahah/sedflow/desiy1/lrg_act',
                         'desiy1.lrg_act.sedflow.modelb.lowz.cdf.grzW1W2.%i.npy' % lrgs['igal'][igal])
    if os.path.isfile(fpost): continue 

    try: 
        print('IGAL = %i' % (lrgs['igal'][igal]))
        with time_limit(30):     
            posterior_samples = desiflow.run(
                    fluxes[igal],           # flux in nmgy
                    sig_fluxes[igal],       # sigma in nmgy 
                    redshift[igal],         # redshift
                    Nsample=n_sample,       # number of samples 
                    progress_bar=False)

            np.save(fpost, posterior_samples)
    except TimeoutException: 
        print('TARGETID = %i TIMEDOUT' % (lrgs['igal'][igal]))
        continue  

