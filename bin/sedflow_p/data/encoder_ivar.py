'''

train encoder for SDSS ivar


'''
import os, sys 

import numpy as np
import torch
from torch import nn
from torch import optim
from accelerate import Accelerator

from sedflow.nns import InfoVAE
import matplotlib.pyplot as plt  

device = torch.device(type='cuda', index=0)

dat_dir = '/scratch/network/chhahn/sedflow/encoder_ivar/'

n_latent = int(sys.argv[1])
i_model  = int(sys.argv[2])

####################################################################
# 1. read SDSS spectra 
####################################################################
wave    = np.load('/scratch/network/chhahn/sedflow/sdss.clean.wave.npy')
ivar    = np.load('/scratch/network/chhahn/sedflow/sdss.clean.ivar.npy')
norm    = np.load('/scratch/network/chhahn/sedflow/sdss.clean.norm_ivar.npy')
mask    = np.load('/scratch/network/chhahn/sedflow/sdss.clean.mask.npy')

# whiten ivar data 
ivar_w  = ivar / norm[:,None]

ivar_zero = (ivar == 0)
avg_ivar = np.array([np.mean(i[~m & ~iz]) for i, m, iz in zip(ivar_w.T, mask.T, ivar_zero.T)])
std_ivar = np.array([np.std(i[~m & ~iz]) for i, m, iz in zip(ivar_w.T, mask.T, ivar_zero.T)])

# last spectral element has no ivar
avg_ivar[-1] = 0. 
avg_ivar[-2] = 0. 
std_ivar[-1] = 1.
std_ivar[-2] = 1.
np.save('/scratch/network/chhahn/sedflow/sdss.clean.ivar.avg.npy', avg_ivar)
np.save('/scratch/network/chhahn/sedflow/sdss.clean.ivar.std.npy', std_ivar)


ivar_w = (ivar_w - avg_ivar[None,:])/std_ivar[None,:]
np.save('/scratch/network/chhahn/sedflow/sdss.clean.ivar_w.npy', ivar_w)

ivar_w  = torch.tensor(ivar_w, dtype=torch.float32)
mask    = torch.tensor(mask, dtype=torch.bool)

Ntrain = int(0.75 * ivar.shape[0])
Nvalid = int(0.15 * ivar.shape[0])

batch_size = 2048
trainloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(ivar_w[:Ntrain][:,None,:], mask[:Ntrain]),
        batch_size=batch_size,
        shuffle=False)

validloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(ivar_w[Ntrain:Ntrain+Nvalid][:,None,:], mask[Ntrain:Ntrain+Nvalid]),
        batch_size=batch_size)

accelerator = Accelerator()
trainloader, validloader = accelerator.prepare(trainloader, validloader)
####################################################################
# train 
####################################################################
label   = f"encoder_ivar.{n_latent}latent.{i_model}"
n_epoch = 1000
nwave   = len(wave)
dropout = 0
lr      = 1e-3

# set architecture
lambd = np.exp(np.random.uniform(0, np.log(1e5)))
    
nhidden0 = int(np.ceil(np.exp(np.random.uniform(np.log(200), np.log(2*nwave)))))
nhidden1 = int(np.ceil(np.exp(np.random.uniform(np.log(100), np.log(nhidden0)))))
nhidden2 = int(np.ceil(np.exp(np.random.uniform(np.log(n_latent+1), np.log(nhidden1)))))
print(f'--- decoder [{nhidden0, nhidden1, nhidden2}] ---')

nkernels = [2,2,2]
npools = [2,2,2]
Nout = nwave - nkernels[0] + 1 
Nout = int((Nout - npools[0])/npools[0] + 1)
Nout = Nout - nkernels[1] + 1
Nout = int((Nout - npools[1])/npools[1] + 1)
Nout = Nout - nkernels[2] + 1
Nout = int((Nout - npools[2])/npools[2] + 1)

nhidden0_enc = int(np.ceil(np.exp(np.random.uniform(np.log(32), np.log(2*Nout)))))
nhidden1_enc = int(np.ceil(np.exp(np.random.uniform(np.log(16), np.log(nhidden0_enc)))))
nhidden2_enc = int(np.ceil(np.exp(np.random.uniform(np.log(n_latent+1), np.log(nhidden1_enc)))))
print(f'--- encoder [{nhidden0_enc, nhidden1_enc, nhidden2_enc}] ---')  

# train model
model = InfoVAE(
    alpha=0, 
    lambd=lambd, 
    nwave=nwave, 
    nkernels=nkernels, 
    npools=npools,
    nhiddens_enc=[nhidden0_enc, nhidden1_enc, nhidden2_enc], 
    nhiddens_dec=[nhidden0, nhidden1, nhidden2],
    ncode=n_latent, 
    dropout=dropout)
model.to(device)
    
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=n_epoch)

best_valid_loss, best_epoch = np.inf, 0
    
losses = []
for epoch in range(n_epoch):
    model.train()
    train_loss = 0.
    for batch in trainloader:
        _ivar, _mask = batch
        
        optimizer.zero_grad()
        loss, _, _, _ = model.loss(_ivar, _mask)
        
        loss.backward()
        train_loss += loss.item() 
        optimizer.step()
        if np.isnan(train_loss): 
            print(_ivar)
            print(_ivar[:,0,:][~_mask])
            print((_ivar[:,0,:][~_mask])[100:])
            print(np.sum(np.isnan(_ivar[:,0,:][~_mask].detach().cpu())))
            raise ValueError
    train_loss /= len(trainloader.dataset)

    with torch.no_grad():
        model.eval()
        valid_loss = 0.
        for batch in validloader:
            _ivar, _mask = batch
            loss, _, _, _ = model.loss(_ivar, _mask)
            valid_loss += loss.item()
        valid_loss /= len(validloader.dataset)

    scheduler.step()
    losses.append((train_loss, valid_loss))

    if epoch % 20 == 0 or epoch == n_epoch - 1:           
        print('====> Epoch: %i TRAINING Loss: %.2e VALIDATION Loss: %.2e' % (epoch, train_loss, valid_loss))

    if valid_loss < best_valid_loss: 
        # save model to file 
        torch.save(model, os.path.join(dat_dir, f'{label}.pt'))
        np.save(os.path.join(dat_dir, f'{label}.losses.npy'), np.array(losses))
        best_valid_loss = valid_loss
        best_epoch = epoch
    #else: 
    #    if epoch > best_epoch + 20: 
    #        break 

# --- validate 
best_model = torch.load(os.path.join(dat_dir, f'{label}.pt'))
best_model.to(device)

w_ivar_recon, _, _ = best_model.forward(torch.tensor(ivar_w[Ntrain+Nvalid:Ntrain+Nvalid+5,None,:], dtype=torch.float32).to(device))
ivar_recon = (w_ivar_recon.detach().cpu().numpy() * std_ivar + avg_ivar) * norm[Ntrain+Nvalid:Ntrain+Nvalid+5][:,None]

fig = plt.figure(figsize=(8, 20))
for i in range(5): 
    sub = fig.add_subplot(5,1,i+1)
    
    _mask = mask[Ntrain+Nvalid+i]
    sub.plot(wave[~_mask], ivar[Ntrain+Nvalid+i][~_mask])
    sub.plot(wave, ivar_recon[i], c='k', ls=':')
    sub.set_xlim(wave[0], wave[-1])
fig.savefig(os.path.join(dat_dir, f'{label}.png'), bbox_inches='tight') 
