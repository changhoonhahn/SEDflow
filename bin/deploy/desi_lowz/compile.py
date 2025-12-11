''' compile posteriors for the DESI LOWZ sample 
'''
import os, sys
import torch
import numpy as np
from astropy.table import Table as aT
from astropy.cosmology import Cosmology, Planck13

from sedflow import flows as F
from sedflow import galaxy as G

if torch.cuda.is_available(): device = 'cuda'
else: device = 'cpu'

# import flows
gsed = G.ModelB(name='modelb.lowz', device=device)
desiflow = F.DESIflow(gsed=gsed, device=device)

lowz = aT.read('/tigress/chhahn/sedflow/alexamon_lowz/LOWZ_z_photometry.csv')


lowz['logmstar_median'] = np.repeat(-999., len(lowz))
lowz['sigma_logmstar'] = np.repeat(-999., len(lowz))
for igal in np.arange(len(lowz)):
    fpost = os.path.join('/tigress/chhahn/sedflow/alexamon_lowz/',
                         'desi_lowz.sedflow.modelb.lowz.cdf.grzW1W2.%i.npy' % igal)
    if os.path.isfile(fpost):
        try:
            post = np.load(fpost)
        except:
            print('%s has issues' % fpost)
        tage = Planck13.age(lowz['Z'][igal]).value # age in Gyr

        # below is to correct for the fact that I forgot to switch log10(Z) to Z
        post[:,7:9] = 10**post[:,7:9]

        # calculate surviving stellar mass
        logmstar = gsed._msurv(post, np.repeat(tage, post.shape[0]))

        lowz['logmstar_median'][igal] = np.median(logmstar)
        lowz['sigma_logmstar'][igal] = np.std(logmstar)
    else:
        print('%s does not exist' % fpost)

    if igal % 5000:
        lowz.write('/tigress/chhahn/sedflow/alexamon_lowz/desi_lowz.sedflow.vagc.hdf5', format='hdf5', overwrite=True)

lowz.write('/tigress/chhahn/sedflow/alexamon_lowz/desi_lowz.sedflow.vagc.hdf5', format='hdf5', overwrite=True)
