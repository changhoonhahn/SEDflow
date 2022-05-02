'''

script to process SDSS data


'''
import os, sys
import numpy as np

import torch

sys.path.append('/home/chhahn/projects/SEDflow/docs/nb/spectra/spectrum-encoder/')
from model import *


def clean_data(): 
    ''' clean up SDSS data for the project and save to file 
    '''
    fsdss = '/scratch/network/chhahn/sedflow/spectra/sdss_spectra.100000.npz'
    data = np.load(fsdss)

    spec = data['spectra']
    wave = 10**data['wave']
    zred = data['z']
    zerr = np.maximum(data['zerr'], 1e-6) # one case has zerr=0, but looks otherwise OK
    ivar = data['ivar']
    mask = data['mask']

    # SDSS IDs
    id = data['plate'], data['mjd'], data['fiber']
    id = [f"{plate}-{mjd}-{fiber}" for plate, mjd, fiber in np.array(id).T]

    # get spectral normalization
    norm_spec = np.nanmedian(spec, axis=1)
    
    wght = ivar * ~mask * (norm_spec**2)[:,None]
    sel = np.any(wght > 0, axis=1)          # remove all spectra that have all zero weights
    sel &= (norm_spec > 0) & (zred < 0.06)  # plus noisy ones and redshift outliers
    sel &= (np.sum(ivar, axis=1) != 0)      # plus ones where ivar is all 0 

    wght = np.maximum(wght, 1e-6)       # avoid zero weights for logL normalization
    
    ivar = ivar[sel]
    wght = wght[sel]
    spec = spec[sel]
    spec_w = spec / norm_spec[sel, None]
    zred = zred[sel]
    zerr = zerr[sel]
    norm_spec = norm_spec[sel]
    mask = mask[sel]
    id = np.array(id)[sel]

    # get ivar normalization 
    zero_ivar = (ivar == 0) 
    norm_ivar = np.array([np.nanmedian(i[~m & ~iz]) for i, m, iz in zip(ivar, mask, zero_ivar)])

    # mask out emission lines 
    w_lines = np.array([
        1033.3  , 1215.67 , 1239.42 , 1305.53 , 1335.52 , 1399.8  ,
        1545.86 , 1640.4  , 1665.85 , 1857.4  , 1908.27 , 2326.   ,
        2439.5  , 2800.32 , 3346.79 , 3426.85 , 3728.3  , 3798.976,
        3836.47 , 3889.   , 3934.777, 3969.588, 4072.3  , 4102.89 ,
        4305.61 , 4341.68 , 4364.436, 4862.68 , 4960.295, 5008.24 ,
        5176.7  , 5895.6  , 6302.046, 6365.536, 6549.86 , 6564.61 ,
        6585.27 , 6707.89 , 6718.29 , 6732.67 ])

    emline_mask = np.zeros(wght.shape).astype(bool)

    for i in range(emline_mask.shape[0]):
        wls = w_lines * (1. + zred[i])
        for wl in wls:
            emline_mask[i] = emline_mask[i] | ((wave > wl - 20) & (wave < wl + 20))
    wght[emline_mask] = 1e-6

    # save to file 
    dat_dir = '/scratch/network/chhahn/sedflow/'
    np.save(os.path.join(dat_dir, 'sdss.clean.wave.npy'), wave)
    np.save(os.path.join(dat_dir, 'sdss.clean.spec.npy'), spec)
    np.save(os.path.join(dat_dir, 'sdss.clean.spec_w.npy'), spec_w)
    np.save(os.path.join(dat_dir, 'sdss.clean.ivar.npy'), ivar)
    np.save(os.path.join(dat_dir, 'sdss.clean.wght.npy'), wght)
    np.save(os.path.join(dat_dir, 'sdss.clean.norm_spec.npy'), norm_spec)
    np.save(os.path.join(dat_dir, 'sdss.clean.norm_ivar.npy'), norm_ivar)
    np.save(os.path.join(dat_dir, 'sdss.clean.mask.npy'), mask)
    np.save(os.path.join(dat_dir, 'sdss.clean.zerr.npy'), zerr)
    np.save(os.path.join(dat_dir, 'sdss.clean.zred.npy'), zred)
    np.save(os.path.join(dat_dir, 'sdss.clean.id.npy'), id)
    return None 


def encode_ivar(n_latent, i_model): 
    ''' encode SDSS ivar using trained autoencoder
    '''
    device = torch.device(type='cuda', index=0)

    # load sdss data 
    wave    = np.load('/scratch/network/chhahn/sedflow/sdss.clean.wave.npy')
    mask    = np.load('/scratch/network/chhahn/sedflow/sdss.clean.mask.npy')
    ivar_w  = np.load('/scratch/network/chhahn/sedflow/sdss.clean.ivar_w.npy')
    avg_ivar = np.load('/scratch/network/chhahn/sedflow/sdss.clean.ivar.avg.npy')
    std_ivar = np.load('/scratch/network/chhahn/sedflow/sdss.clean.ivar.std.npy')

    # load best ivar model
    best_model = torch.load(os.path.join('/scratch/network/chhahn/sedflow/encoder_ivar/',
        f"encoder_ivar.{n_latent}latent.{i_model}.pt"))
    best_model.to(device)

    _mu, _logvar = best_model.encode(torch.tensor(ivar_w[:,None,:], dtype=torch.float32).to(device))
    h = best_model.reparameterize(_mu, _logvar).cpu().detach().numpy()

    np.save('/scratch/network/chhahn/sedflow/sdss.clean.h_ivar.npy', h)
    return None 


def encode_spec(n_latent, i_model): 
    ''' encode SDSS spec using trained autoencoder
    '''
    device = torch.device(type='cuda', index=0)

    # load SDSS data
    wave    = np.load('/scratch/network/chhahn/sedflow/sdss.clean.wave.npy')
    spec_w  = np.load('/scratch/network/chhahn/sedflow/sdss.clean.spec_w.npy')
    wght    = np.load('/scratch/network/chhahn/sedflow/sdss.clean.wght.npy')
    zred    = np.load('/scratch/network/chhahn/sedflow/sdss.clean.zred.npy')
    
    spec_w  = torch.tensor(spec_w, dtype=torch.float32).to(device)
    wght    = torch.tensor(wght, dtype=torch.float32).to(device)
    zred    = torch.tensor(zred, dtype=torch.float32).to(device)

    # laod best model model
    best_model = torch.load('/scratch/network/chhahn/sedflow/encoder_spec/encoder_spec.emline_mask.%ilatent.%i.pt' % (n_latent, i_model))
    best_model.to(device)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(spec_w, wght, zred), 
        batch_size=512, 
        shuffle=False)

    # define SDSS instrument
    sdss = Instrument(torch.tensor(wave, dtype=torch.float32).to(device))

    h_spec = []
    for batch in loader: 
        spec, w, z = batch

        h, _, _ = best_model._forward(spec, w, instrument=sdss, z=z)

        h_spec.append(h.detach().cpu())
    h_spec = np.concatenate(h_spec, axis=0)
    np.save('/scratch/network/chhahn/sedflow/sdss.clean.h_spec.npy', h_spec)
    return None 


def best_encoder_ivar(n_latent): 
    '''
    '''
    dat_dir = '/scratch/network/chhahn/sedflow/encoder_ivar/'
    valid_losses = [] 
    for i_model in range(10): 

        label   = f"encoder_ivar.{n_latent}latent.{i_model}"
        losses  = np.load(os.path.join(dat_dir, f'{label}.losses.npy'))
        valid_losses.append(np.min(losses[:,1])) 
    print(np.argmin(valid_losses))
    return None 


def best_encoder_spec(n_latent): 
    '''
    '''
    dat_dir = '/scratch/network/chhahn/sedflow/encoder_spec/'
    valid_losses = [] 
    for i_model in range(10): 
        label   = f"encoder_spec.emline_mask.{n_latent}latent.{i_model}"
        losses  = np.load(os.path.join(dat_dir, f'{label}.losses.npy'))
        print(i_model, np.min(losses[:,1]))
        valid_losses.append(np.min(losses[:,1])) 
    print(f'>>>{np.argmin(valid_losses)}')
    return None 


def train_nde_noise(n_model=5):
    ''' train NDE to estimate p(A_ivar, h_ivar | A_spec, z) from SDSS data 
    '''
    from sbi import utils as Ut
    from sbi import inference as Inference
    from torch.utils.tensorboard.writer import SummaryWriter

    dat_dir = '/scratch/network/chhahn/sedflow/nde_noise/'

    # load training SDSS data 
    A_ivar  = np.load('/scratch/network/chhahn/sedflow/sdss.clean.norm_ivar.npy')[:,None]
    h_ivar  = np.load('/scratch/network/chhahn/sedflow/sdss.clean.h_ivar.npy')
    A_spec  = np.load('/scratch/network/chhahn/sedflow/sdss.clean.norm_spec.npy')[:,None]
    zred    = np.load('/scratch/network/chhahn/sedflow/sdss.clean.zred.npy')[:,None]

    y_train = np.concatenate([A_ivar, h_ivar], axis=1)
    x_train = np.concatenate([np.log10(A_spec.clip(1e-2, None)), zred], axis=1)

    # set prior 
    lower_bounds = torch.tensor(np.min(y_train, axis=0))
    upper_bounds = torch.tensor(np.max(y_train, axis=0))
    prior = Ut.BoxUniform(low=lower_bounds, high=upper_bounds, device='cuda')

    anpes, phats, archs = [], [], []
    valid_logprobs, best_valid_logprobs = [], []
    for i in range(n_model): 
        nhidden = int(np.ceil(np.exp(np.random.uniform(np.log(64), np.log(512)))))
        nblocks = int(np.random.uniform(3, 10))
        print('MAF with nhidden=%i; nblocks=%i' % (nhidden, nblocks))
        archs.append('%ix%i' % (nhidden, nblocks))

        anpe = Inference.SNPE(prior=prior, 
                density_estimator=Ut.posterior_nn('maf', hidden_features=nhidden, num_transforms=nblocks), 
                device='cuda',
                summary_writer=SummaryWriter(os.path.join(dat_dir,
                    'nde.noise.%ix%i.%i' % (nhidden, nblocks, i))))
        anpe.append_simulations(
            torch.as_tensor(y_train.astype(np.float32)), 
            torch.as_tensor(x_train.astype(np.float32)))

        phat = anpe.train()

        nde = anpe.build_posterior(phat) 

        fanpe = os.path.join(dat_dir, f'nde.noise.{nhidden}x{nblocks}.pt')
        torch.save(nde, fanpe)
        np.save(fanpe.replace('.pt', '.valid_loss.npy'), np.array(anpe._summary['validation_log_probs']))

        anpes.append(anpe)
        phats.append(phat)

        valid_logprobs.append(anpe._summary['validation_log_probs'])
        best_valid_logprobs.append(anpe._summary['best_validation_log_probs'])

    ibest = np.argmax(best_valid_logprobs)
    best_anpe = anpes[ibest]
    best_phat = phats[ibest]
    best_arch = archs[ibest]
    
    best_nde = best_anpe.build_posterior(best_phat) 

    #save trained ANPE
    fanpe = os.path.join(dat_dir, 'nde.noise.best.pt')
    print('saving to %s' % fanpe)
    torch.save(best_nde, fanpe)
    np.save(fanpe.replace('.pt', '.valid_loss.npy'), np.array(valid_logprobs[ibest]))
    return None 


def decode_ivar(): 
    ''' use best autoencoder for IVAR to decode IVAR latent variables 
    '''
    from torch import nn
    from torch.autograd import Variable
    from torch.nn import functional as F

    device = torch.device(type='cuda', index=0)
    
    dat_dir = '/scratch/network/chhahn/sedflow/spectra/'
    
    # load ivar autoencoder model 
    ibest = 6 # this is hardcoded. don't touch
    state_dict = torch.load(os.path.join(dat_dir, 'ivar.vae_model.%i.pt' % ibest))
    best_model = InfoVAE(alpha=0, lambd=0, # these don't matter here
                    nwave=nwave,  
                    ncode=nlatent, 
                    nkernels=[state_dict['conv0.weight'].shape[-1], state_dict['conv1.weight'].shape[-1], state_dict['conv2.weight'].shape[-1]],
                    nhiddens_enc=[state_dict['enc0.weight'].shape[0], state_dict['enc1.weight'].shape[0], state_dict['enc2.weight'].shape[0]], 
                    nhiddens_dec=[state_dict['decd.weight'].shape[0], state_dict['decd2.weight'].shape[0], state_dict['decd3.weight'].shape[0]], 
                    dropout=dropout)
    best_model.load_state_dict(state_dict) 
    best_model.to(device)

    # load redshift, latent variables, and normalization 
    zred    = np.load(os.path.join(dat_dir, 'ivar.vae_model.6.zred.npy'))
    h_ivar  = np.load(os.path.join(dat_dir, 'ivar.vae_model.6.latvar.npy')) # ivar latent variables 
    A_ivar  = np.load(os.path.join(dat_dir, 'ivar.vae_model.6.ivar_norm.npy')) # ivar noramlization 

    # decode latent variables
    ivar_recon = best_model.decode(torch.tensor(h_ivar).to(device)).detach().cpu().numpy()
    
    # reconstruct  
    avg_ivar = np.load(os.path.join(dat_dir, 'ivar.avg.npy'))
    std_ivar = np.load(os.path.join(dat_dir, 'ivar.std.npy'))

    ivar = (std_ivar * ivar_recon + avg_ivar) * A_ivar

    np.save('/scratch/network/chhahn/sedflow/sedflow_p.obs.v0.1.ivar.decoded.npy',
            ivar) 
    return None 


if __name__=='__main__': 
    task = sys.argv[1]

    if task == 'clean_data': 
        clean_data()
    elif task == 'encode_ivar': 
        n_latent    = int(sys.argv[2])
        i_model     = int(sys.argv[3])
        encode_ivar(n_latent, i_model)
    elif task == 'encode_spec': 
        n_latent    = int(sys.argv[2])
        i_model     = int(sys.argv[3])
        encode_spec(n_latent, i_model)
    elif task == 'best_encoder_ivar': 
        n_latent    = int(sys.argv[2])
        best_encoder_ivar(n_latent)
    elif task == 'best_encoder_spec': 
        n_latent    = int(sys.argv[2])
        best_encoder_spec(n_latent)
    elif task == 'train_nde_noise': 
        train_nde_noise()
