'''

a script for adding noise to the SEDs to generate spectra. The nosie mode
is an NDE 

'''
import os, sys
import numpy as np 

from provabgs import util as UT
from sedflow import nns as NNs

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from sbi import utils as Ut
from sbi import inference as Inference


dat_dir = '/scratch/network/chhahn/sedflow/training_sed/'

# load noiseless SEDs
ibatch = int(sys.argv[1])

wave = np.load(os.path.join(dat_dir, f'train.v0.1.{ibatch}.waves.npy'))
seds = np.load(os.path.join(dat_dir, f'train.v0.1.{ibatch}.seds.npy'))

# read in SDSS wavelength 
rawspec = np.load(os.path.join('/scratch/network/chhahn/sedflow/spectra/', 'sdss_spectra.100000.npz')) # SDSS dr16 spectra
w_sdss = 10**rawspec['wave']

# read in (h, Aivar) values 
h_Aivar = np.zeros((seds.shape[0], 6))
for isplit in range(10): 
    _hA = np.load(os.path.join('/scratch/network/chhahn/sedflow/spectra/', 'h_Aivar.nde.%i.%iof10.npy' % (ibatch, isplit+1)))
    h_Aivar[isplit::10] = _hA
np.save(os.path.join('/scratch/network/chhahn/sedflow/spectra/', 'h_Aivar.nde.%i.npy' % (ibatch)), h_Aivar)

# load in ivar decoder 
i_best = 6
vae_dict = torch.load(os.path.join('/scratch/network/chhahn/sedflow/spectra/',
    f'ivar.vae_model.{i_best}.pt'), map_location=torch.device('cpu'))

nhiddens_enc = [vae_dict['enc0.weight'].shape[0], vae_dict['enc1.weight'].shape[0], vae_dict['enc2.weight'].shape[0]]
nhiddens_dec = nhiddens_enc # accidentally set them as the same

avg_ivar = np.load(os.path.join('/scratch/network/chhahn/sedflow/spectra/', 'ivar.avg.npy'))
std_ivar = np.load(os.path.join('/scratch/network/chhahn/sedflow/spectra/', 'ivar.std.npy'))

model = NNs.InfoVAE(
        alpha=0,
        lambd=0,
        nwave=len(w_sdss),
        nkernels=[2, 2, 2],
        npools=[2, 2, 2],
        nhiddens_enc=nhiddens_enc,
        nhiddens_dec=nhiddens_dec,
        ncode=5,
        dropout=0)
model.load_state_dict(vae_dict)

# use decoder to calculate whitened ivar 
w_ivar = model.decode(torch.tensor(h_Aivar[:,:-1].astype(np.float32))).detach().cpu().numpy()
# calculate ivar 
ivar = (w_ivar * std_ivar + avg_ivar) * h_Aivar[:,-1][:,None]

# bin into SDSS wavelength and apply noise model 
seds_sdss = np.empty((seds.shape[0], len(w_sdss)))
for i in range(seds.shape[0]): 
    seds_sdss[i,:] = UT.trapz_rebin(wave[i], seds[i], w_sdss) + ivar[i]**-0.5 * np.random.normal(size=len(w_sdss))

np.save(os.path.join(dat_dir, f'train.v0.1.{ibatch}.ivar.vae_noise.npy'), ivar) 
np.save(os.path.join(dat_dir, f'train.v0.1.{ibatch}.seds.vae_noise.npy'), seds_sdss)
