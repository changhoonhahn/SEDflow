'''

script to encode SDSS spectra and postporcess some things 

'''
import os, sys
import numpy as np

import torch

sys.path.append('/home/chhahn/projects/SEDflow/docs/nb/spectra/spectrum-encoder/')
from model import *

device = torch.device(type='cuda', index=0)

################################################################
# load in SDSS spectra
################################################################
dat_dir = '/scratch/network/chhahn/sedflow/spectra/'
rawspec = np.load(os.path.join(dat_dir, 'sdss_spectra.100000.npz')) # SDSS dr16 spectra
bad_ivar = (np.sum(rawspec['ivar'], axis=1) == 0)
zlim = (rawspec['z'] < 0.1)
cuts = zlim & ~bad_ivar

wave = 10**rawspec['wave']
spec = rawspec['spectra'][cuts]
ivar = rawspec['ivar'][cuts]
mask = rawspec['mask'][cuts]
zred = rawspec['z'][cuts]

norm = get_norm(spec)

_zred = np.load(os.path.join(dat_dir, 'ivar.vae_model.6.zred.npy'))
assert np.array_equal(zred, _zred.flatten())

h_ivar = np.load(os.path.join(dat_dir, 'ivar.vae_model.6.latvar.npy')) # ivar latent variables 
A_ivar = np.load(os.path.join(dat_dir, 'ivar.vae_model.6.ivar_norm.npy')) # ivar noramlization 

################################################################
# some postprocessing 
################################################################
sdss_w = ivar * ~mask * (norm**2)[:,None]
sel = np.any(sdss_w > 0, axis=1)   # remove all spectra that have all zero weights
sel &= (norm > 0) & (zred < 0.5)   # plus noisy ones and redshift outliers

sdss_w = np.maximum(sdss_w, 1e-6)       # avoid zero weights for logL normalization
sdss_w = sdss_w[sel]
sdss_y = spec[sel] / norm[sel, None]
sdss_z = zred[sel]
sdss_norm = norm[sel]

sdss_h_ivar = h_ivar[sel]
sdss_A_ivar = A_ivar[sel]

sdss_w = torch.tensor(sdss_w).to(device)
sdss_y = torch.tensor(sdss_y).to(device)
sdss_z = torch.tensor(sdss_z).to(device)
sdss_norm = torch.tensor(sdss_norm).to(device)

################################################################
# load in best encoder model
################################################################
best_model = torch.load('/home/chhahn/projects/SEDflow/docs/nb/spectra/spectrum-encoder/sedflow.specoder.emline_mask.0.pt')
best_model.to(device)

# define SDSS instrument
wave_obs = torch.tensor(np.load('/home/chhahn/projects/SEDflow/docs/nb/spectra/spectrum-encoder/wave_sdss.npy'))
sdss = Instrument(wave_obs.to(device))

# mask out emission lines OII, Hbeta, OIII, Halpha
emline_mask = np.zeros(sdss_w.shape).astype(bool)

w_lines = np.array([1033.3  , 1215.67 , 1239.42 , 1305.53 , 1335.52 , 1399.8  ,
       1545.86 , 1640.4  , 1665.85 , 1857.4  , 1908.27 , 2326.   ,
       2439.5  , 2800.32 , 3346.79 , 3426.85 , 3728.3  , 3798.976,
       3836.47 , 3889.   , 3934.777, 3969.588, 4072.3  , 4102.89 ,
       4305.61 , 4341.68 , 4364.436, 4862.68 , 4960.295, 5008.24 ,
       5176.7  , 5895.6  , 6302.046, 6365.536, 6549.86 , 6564.61 ,
       6585.27 , 6707.89 , 6718.29 , 6732.67 ])
for i in range(emline_mask.shape[0]):
    wls = w_lines * (1. + sdss_z[i].cpu().numpy())
    for wl in wls:
        emline_mask[i] = emline_mask[i] | ((wave_obs.cpu().numpy() > wl - 40) & (wave_obs.cpu().numpy() < wl + 40))
sdss_w[emline_mask] = 1e-6
sdss_w = sdss_w.clip(1e-6, None)

sdss_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(sdss_y, sdss_w, sdss_z),
    batch_size=512) 

with torch.no_grad():
    best_model.eval()

    s_sdss = []
    for batch in sdss_loader:
        spec, w, z = batch
        s, _, _ = best_model._forward(spec, w, instrument=sdss, z=z)
        s_sdss.append(s.detach().cpu())
s_sdss = np.concatenate(s_sdss, axis=0)

# save normalizati and latent variables
np.save(os.path.join('/scratch/network/chhahn/sedflow/', f'sedflow_p.obs.v0.1.encoded.npy'),
        np.concatenate([sdss_norm.cpu()[:,None], s_sdss], axis=1))
np.save(os.path.join('/scratch/network/chhahn/sedflow/', f'sedflow_p.obs.v0.1.ivar.encoded.npy'),
        np.concatenate([sdss_A_ivar, sdss_h_ivar], axis=1))
np.save(os.path.join('/scratch/network/chhahn/sedflow/', f'sedflow_p.obs.v0.1.zred.npy'),
        sdss_z.cpu().numpy())
np.save('/scratch/network/chhahn/sedflow/sedflow_p.obs.v0.1.spectra.npy',
        rawspec['spectra'][cuts][sel])
np.save('/scratch/network/chhahn/sedflow/sedflow_p.obs.v0.1.ivar.npy',
        rawspec['ivar'][cuts][sel])
np.save('/scratch/network/chhahn/sedflow/sedflow_p.obs.v0.1.mask.npy',
        rawspec['mask'][cuts][sel])
print(rawspec['wave'].shape)
np.save('/scratch/network/chhahn/sedflow/sedflow_p.obs.v0.1.log_wave.npy', 
        rawspec['wave'])
