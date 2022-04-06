'''

a script for adding noise to the SEDs to generate spectra. The nosie mode
is an NDE 

'''
import os, sys
import numpy as np 

import torch
from sbi import utils as Ut
from sbi import inference as Inference


dat_dir = '/scratch/network/chhahn/sedflow/training_sed/'

# load noiseless SEDs
ibatch = int(sys.argv[1])
isplit = int(sys.argv[2])

wave = np.load(os.path.join(dat_dir, f'train.v0.1.{ibatch}.waves.npy'))
seds = np.load(os.path.join(dat_dir, f'train.v0.1.{ibatch}.seds.npy'))[isplit::10]
zred = np.load(os.path.join(dat_dir, f'train.v0.1.{ibatch}.redshifts.npy'))[isplit::10]
theta = np.load(os.path.join(dat_dir, f'train.v0.1.{ibatch}.thetas_sps.npy'))[isplit::10]

# read in SDSS spectra
rawspec = np.load(os.path.join('/scratch/network/chhahn/sedflow/spectra/', 'sdss_spectra.100000.npz')) # SDSS dr16 spectra
w_sdss = 10**rawspec['wave']

wlim = (wave > w_sdss[0]) & (wave < w_sdss[-1])

A_spec = np.array([np.median(sed[_w]) for sed, _w in zip(seds, wlim)])


# assign h, Aivar 
dat_dir = '/scratch/network/chhahn/sedflow/spectra/'
i_best = 6
h_ivar      = np.load(os.path.join(dat_dir, f'ivar.vae_model.{i_best}.latvar.npy'))
A_ivar      = np.load(os.path.join(dat_dir, f'ivar.vae_model.{i_best}.ivar_norm.npy'))
A_spec_vae  = np.load(os.path.join(dat_dir, f'ivar.vae_model.{i_best}.spec_norm.npy'))
zred_vae    = np.load(os.path.join(dat_dir, f'ivar.vae_model.{i_best}.zred.npy'))

N_train = 1000
y_train = np.concatenate([h_ivar, A_ivar], axis=1)[N_train:]
x_train = np.concatenate([np.log10(A_spec_vae.clip(1e-2, None)), zred_vae], axis=1)[N_train:]

# read in NDE 
lower_bounds = torch.tensor([-4, -10, -7, -5, -16, 0])
upper_bounds = torch.tensor([10, 3.5, 7, 5, 5.5, 17])

prior = Ut.BoxUniform(low=lower_bounds, high=upper_bounds, device='cpu')

# load NDE
best_arch = '279x8'
nhidden, nblocks = 279, 8
fanpe = os.path.join(dat_dir, f'sdss_ivar.noise_nde.{best_arch}.pt')

anpe = Inference.SNPE(prior=prior,
                      density_estimator=Ut.posterior_nn('maf', hidden_features=nhidden, num_transforms=nblocks),
                      device='cpu')
# load ANPE
anpe.append_simulations(
    torch.as_tensor(y_train.astype(np.float32)),
    torch.as_tensor(x_train.astype(np.float32)))

phat = anpe._build_neural_net(torch.as_tensor(y_train.astype(np.float32)), torch.as_tensor(x_train.astype(np.float32)))
phat.load_state_dict(torch.load(fanpe, map_location=torch.device('cpu')))

anpe._x_shape = Ut.x_shape_from_simulation(torch.as_tensor(y_train.astype(np.float32)))

best_phat = anpe.build_posterior(phat)

# assign h, A_ivar
h_Aivar = []
for i in range(len(zred)):
    _y = best_phat.sample((1,), x=torch.as_tensor(np.array([np.log10(A_spec[i].clip(1e-2, None)), zred[i]]).flatten().astype(np.float32)).to('cpu'), 
            show_progress_bars=False)
    h_Aivar.append(np.array(_y.detach().to('cpu')[0]))
h_Aivar = np.array(h_Aivar)

np.save(os.path.join('/scratch/network/chhahn/sedflow/spectra/', 'h_Aivar.nde.%iof10.npy' % (isplit+1)), h_Aivar)
