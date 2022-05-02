'''

script train spectral encoder 

'''
import os, sys
import numpy as np

import torch
from torch import nn
from torch import optim
from accelerate import Accelerator

sys.path.append('/home/chhahn/projects/SEDflow/docs/nb/spectra/spectrum-encoder/')
from model import *

import matplotlib.pyplot as plt  
device = torch.device(type='cuda', index=0)

dat_dir = '/scratch/network/chhahn/sedflow/encoder_spec/'

n_latent = int(sys.argv[1])
i_model  = int(sys.argv[2])

####################################################################
# 1. read SDSS spectra 
####################################################################
wave    = np.load('/scratch/network/chhahn/sedflow/sdss.clean.wave.npy')
spec_w  = np.load('/scratch/network/chhahn/sedflow/sdss.clean.spec_w.npy')
wght    = np.load('/scratch/network/chhahn/sedflow/sdss.clean.wght.npy')
zred    = np.load('/scratch/network/chhahn/sedflow/sdss.clean.zred.npy')

spec_w  = torch.tensor(spec_w.astype(np.float32))
wght    = torch.tensor(wght.astype(np.float32))
zred    = torch.tensor(zred.astype(np.float32))
####################################################################
# train 
####################################################################
Ntrain = int(0.75 * spec_w.shape[0])
Nvalid = int(0.15 * spec_w.shape[0])
batch_size=512
trainloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(spec_w[:Ntrain], wght[:Ntrain], zred[:Ntrain]),
        batch_size=batch_size,
        shuffle=False)

validloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(spec_w[Ntrain:Ntrain+Nvalid], wght[Ntrain:Ntrain+Nvalid], zred[Ntrain:Ntrain+Nvalid]),
        batch_size=batch_size)


# restframe wavelength for reconstructed spectra
lmbda_min = wave.min()/(1+zred.max())
lmbda_max = wave.max()
bins = int(wave.shape[0] * (1 + zred.max()))
wave_rest = torch.linspace(lmbda_min, lmbda_max, bins, dtype=torch.float32)
np.save(os.path.join(dat_dir, 'encoder_spec.wave_rest.npy'), wave_rest)

print ("Observed frame:\t{:.0f} .. {:.0f} A ({} bins)".format(wave.min(), wave.max(), len(wave)))
print ("Restframe:\t{:.0f} .. {:.0f} A ({} bins)".format(lmbda_min, lmbda_max, bins))

# define SDSS instrument
sdss = Instrument(torch.tensor(wave, dtype=torch.float32))


def train(model, accelerator, instrument, trainloader, validloader,
        n_epoch=200, label="", silent=False, lr=3e-4, patience=20):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=n_epoch)
    model, optimizer = accelerator.prepare(model, optimizer)
    
    best_valid_loss, best_epoch = np.inf, 0 
    losses = []
    for epoch in range(n_epoch):
        model.train()
        train_loss = 0.
        for batch in trainloader:
            spec, w, z = batch
            optimizer.zero_grad()
            loss = model.loss(spec, w, instrument=instrument, z=z)
            accelerator.backward(loss)
            train_loss += loss.item()
            optimizer.step()
        train_loss /= len(trainloader.dataset)

        with torch.no_grad():
            model.eval()
            valid_loss = 0.
            for batch in validloader:
                spec, w, z = batch
                loss = model.loss(spec, w, instrument=instrument, z=z)
                valid_loss += loss.item()
            valid_loss /= len(validloader.dataset)

        scheduler.step()
        losses.append((train_loss, valid_loss))

        if epoch % 20 == 0 or epoch == n_epoch - 1:
            if not silent:
                print('====> Epoch: %i TRAINING Loss: %.2e VALIDATION Loss: %.2e' % (epoch, train_loss, valid_loss))
            
        if valid_loss < best_valid_loss: 
            torch.save(model, os.path.join(dat_dir, f'{label}.pt'))
            np.save(os.path.join(dat_dir, f'{label}.losses.npy'), np.array(losses))
            best_valid_loss = valid_loss
            best_epoch = epoch
        #else: 
        #    if epoch > best_epoch + patience: 
        #        break 
    return None 

label = f"encoder_spec.emline_mask.{n_latent}latent.{i_model}"
n_epoch = 400

accelerator = Accelerator()
trainloader, validloader, sdss = accelerator.prepare(trainloader, validloader, sdss)

model = SpectrumAutoencoder(wave_rest, n_latent=n_latent)

train(model, accelerator, sdss, trainloader, validloader, n_epoch=n_epoch, label=label, lr=1e-3)

# --- validate 
best_model = torch.load(os.path.join(dat_dir, f'{label}.pt'))
best_model.to(device)


fig = plt.figure(figsize=(8, 20))
for i in range(5): 
    sub = fig.add_subplot(5,1,i+1)

    _spec = torch.tensor(spec_w[Ntrain+Nvalid:][i:i+1], dtype=torch.float32).to(device)
    _wght = torch.tensor(wght[Ntrain+Nvalid:][i:i+1], dtype=torch.float32).to(device)
    _zred = torch.tensor(zred[Ntrain+Nvalid:][i:i+1], dtype=torch.float32).to(device)
    
    s, f_rest, f_obs = best_model._forward(_spec, _wght, instrument=sdss, z=_zred)
    
    sub.plot(wave, f_obs.detach().cpu().flatten(), lw=1)
    sub.plot(wave_rest.detach().cpu().numpy() * (1. + _zred.detach().cpu().numpy()), f_rest.detach().cpu().flatten(), c='k', lw=0.5, ls=':')
    sub.set_xlim(wave_rest.detach().cpu()[0], wave_rest.detach().cpu()[-1])
fig.savefig(os.path.join(dat_dir, f'{label}.png'), bbox_inches='tight') 
