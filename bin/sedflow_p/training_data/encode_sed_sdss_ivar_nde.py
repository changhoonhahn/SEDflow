import os, sys
import numpy as np

import torch

sys.path.append('/home/chhahn/projects/SEDflow/docs/nb/spectra/spectrum-encoder/')
from model import *


ibatch = int(sys.argv[1]) 


device = torch.device(type='cuda', index=0)

# load in best encoder model
best_model = torch.load('/home/chhahn/projects/SEDflow/docs/nb/spectra/spectrum-encoder/sedflow.specoder.emline_mask.0.pt')
best_model.to(device)

# load in training data 
train_dir = '/scratch/network/chhahn/sedflow/training_sed/'
train_seds = np.load(os.path.join(train_dir,
    'train.v0.1.%i.seds.vae_noise.npy' % ibatch)).astype(np.float32)
train_ivar = np.load(os.path.join(train_dir, 
    'train.v0.1.%i.ivar.vae_noise.npy' % ibatch)).astype(np.float32)
train_zred = np.load(os.path.join(train_dir, 
    'train.v0.1.%i.redshifts.npy' % ibatch)).astype(np.float32)


# define SDSS instrument
wave_obs = torch.tensor(np.load('/home/chhahn/projects/SEDflow/docs/nb/spectra/spectrum-encoder/wave_sdss.npy'))
sdss = Instrument(wave_obs.to(device))

# mask nonsense 
train_mask = ~np.isfinite(train_seds)
train_seds[train_mask] = 0.

# get nomralization 
train_norm = get_norm(train_seds)

train_w = torch.tensor(train_ivar * ~train_mask * (train_norm**2)[:,None]).to(device)
train_y = torch.tensor(train_seds / train_norm[:, None]).to(device)
train_z = torch.tensor(train_zred).to(device)
train_norm = torch.Tensor(train_norm).to(device)

# mask out emission lines OII, Hbeta, OIII, Halpha
emline_mask_train = np.zeros(train_w.shape).astype(bool)

w_lines = np.array([1033.3  , 1215.67 , 1239.42 , 1305.53 , 1335.52 , 1399.8  ,
       1545.86 , 1640.4  , 1665.85 , 1857.4  , 1908.27 , 2326.   ,
       2439.5  , 2800.32 , 3346.79 , 3426.85 , 3728.3  , 3798.976,
       3836.47 , 3889.   , 3934.777, 3969.588, 4072.3  , 4102.89 ,
       4305.61 , 4341.68 , 4364.436, 4862.68 , 4960.295, 5008.24 ,
       5176.7  , 5895.6  , 6302.046, 6365.536, 6549.86 , 6564.61 ,
       6585.27 , 6707.89 , 6718.29 , 6732.67 ])
for i in range(emline_mask_train.shape[0]):
    wls = w_lines * (1. + train_z[i].cpu().numpy())
    for wl in wls:
        emline_mask_train[i] = emline_mask_train[i] | ((wave_obs.cpu().numpy() > wl - 40) & (wave_obs.cpu().numpy() < wl + 40))
train_w[emline_mask_train] = 1e-6
train_w = train_w.clip(1e-6, None)


train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_y, train_w, train_z),
    batch_size=512) 

with torch.no_grad():
    best_model.eval()

    s_train = []
    for batch in train_loader:
        spec, w, z = batch
        s, _, _ = best_model._forward(spec, w, instrument=sdss, z=z)
        s_train.append(s.detach().cpu())
s_train = np.concatenate(s_train, axis=0)

# save normalizati and latent variables
np.save(os.path.join(train_dir, f'train.v0.1.{ibatch}.encoded.npy'),
        np.concatenate([train_norm.cpu()[:,None], s_train], axis=1))
