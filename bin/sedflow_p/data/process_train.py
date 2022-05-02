'''

scripts for processing the training data for SEDflow+

'''
import os, sys
import numpy as np 

import torch


def sample_nde_noise(ibatch, isplit, best_arch): 
    ''' sample the NDE estimate of p(A_ivar, h_ivar | A_spec, z) 
    '''
    from torch.utils.tensorboard.writer import SummaryWriter

    from sbi import utils as Ut
    from sbi import inference as Inference

    # load noiseless SEDs
    dat_dir = '/scratch/network/chhahn/sedflow/training_sed/'
    wave = np.load(os.path.join(dat_dir, f'train.v0.1.{ibatch}.waves.npy'))
    seds = np.load(os.path.join(dat_dir, f'train.v0.1.{ibatch}.seds.npy'))[isplit::10]
    zred = np.load(os.path.join(dat_dir, f'train.v0.1.{ibatch}.redshifts.npy'))[isplit::10]

    # get estimate of training spectra normalization
    w_sdss = np.load('/scratch/network/chhahn/sedflow/sdss.clean.wave.npy')
    wlim = (wave > w_sdss[0]) & (wave < w_sdss[-1])
    A_spec = np.array([np.median(sed[_w]) for sed, _w in zip(seds, wlim)])
    
    # load NDE
    dat_dir = '/scratch/network/chhahn/sedflow/nde_noise/'
    fanpe = os.path.join(dat_dir, f'nde.noise.{best_arch}.pt')
    best_phat = torch.load(fanpe, map_location=torch.device('cpu'))
    best_phat._device = torch.device('cpu')

    # sample A_ivar, h_ivar
    Ah_ivar = []
    for i in range(len(zred)):
        _y = best_phat.sample((1,), 
                x=torch.as_tensor(np.array([
                    np.log10(A_spec[i].clip(1e-2, None)), 
                    zred[i]]).flatten(), dtype=torch.float32).to('cpu'), 
                show_progress_bars=False)
        Ah_ivar.append(np.array(_y.detach().to('cpu')[0]))
    Ah_ivar = np.array(Ah_ivar)

    np.save(os.path.join('/scratch/network/chhahn/sedflow/nde_noise/',
        'Ah_ivar.nde.%i.%iof10.npy' % (ibatch, isplit+1)), Ah_ivar)
    return None 


def apply_nde_noise(ibatch, n_latent, i_model): 
    '''
    '''
    from provabgs import util as UT
    from sedflow import nns as NNs

    dat_dir = '/scratch/network/chhahn/sedflow/training_sed/'

    # load noiseless SEDs
    wave = np.load(os.path.join(dat_dir, f'train.v0.1.{ibatch}.waves.npy'))
    seds = np.load(os.path.join(dat_dir, f'train.v0.1.{ibatch}.seds.npy'))

    # read in SDSS wavelength 
    w_sdss = np.load('/scratch/network/chhahn/sedflow/sdss.clean.wave.npy')

    # read in (A_ivar, h_ivar) values 
    f_ah = os.path.join('/scratch/network/chhahn/sedflow/nde_noise/',
            'Ah_ivar.nde.%i.npy' % ibatch)
    if not os.path.isfile(f_ah): 
        Ah_ivar = np.zeros((seds.shape[0], 6))
        for isplit in range(10): 
            _Ah = np.load(os.path.join('/scratch/network/chhahn/sedflow/nde_noise/',
                'Ah_ivar.nde.%i.%iof10.npy' % (ibatch, isplit+1)))
            Ah_ivar[isplit::10] = _Ah
        np.save(f_ah, Ah_ivar)
    else: 
        Ah_ivar = np.load(f_ah)

    # load in ivar decoder 
    model = torch.load(os.path.join('/scratch/network/chhahn/sedflow/encoder_ivar/',
        f"encoder_ivar.{n_latent}latent.{i_model}.pt"), 
        map_location=torch.device('cpu'))
    avg_ivar = np.load('/scratch/network/chhahn/sedflow/sdss.clean.ivar.avg.npy')
    std_ivar = np.load('/scratch/network/chhahn/sedflow/sdss.clean.ivar.std.npy')

    # use decoder to calculate whitened ivar 
    w_ivar = model.decode(
            torch.tensor(Ah_ivar[:,1:], dtype=torch.float32)
            ).detach().cpu().numpy()
    # calculate ivar 
    ivar = (w_ivar * std_ivar + avg_ivar) * Ah_ivar[:,0][:,None]
    norm_ivar = np.nanmedian(ivar, axis=1)

    # bin into SDSS wavelength and apply noise model 
    seds_sdss = np.empty((seds.shape[0], len(w_sdss)))
    for i in range(seds.shape[0]): 
        seds_sdss[i,:] = UT.trapz_rebin(wave[i], seds[i], w_sdss) + ivar[i]**-0.5 * np.random.normal(size=len(w_sdss))
    
    np.save(os.path.join(dat_dir, f'train.v0.1.{ibatch}.h_ivar.nde_noise.npy'), 
            Ah_ivar[:,1:]) 
    np.save(os.path.join(dat_dir, f'train.v0.1.{ibatch}.norm_ivar.nde_noise.npy'), 
            norm_ivar) 
    np.save(os.path.join(dat_dir, f'train.v0.1.{ibatch}.ivar.nde_noise.npy'), 
            ivar) 
    np.save(os.path.join(dat_dir, f'train.v0.1.{ibatch}.seds.nde_noise.npy'), 
            seds_sdss)
    return None


def encode_spec(ibatch, n_latent, i_model): 
    ''' encode trianing spec using trained autoencoder
    '''
    sys.path.append('/home/chhahn/projects/SEDflow/docs/nb/spectra/spectrum-encoder/')
    from model import Instrument

    device = torch.device('cpu') 
    # load training data 
    dat_dir = '/scratch/network/chhahn/sedflow/training_sed/'
    train_seds = np.load(os.path.join(dat_dir, 
        f'train.v0.1.{ibatch}.seds.nde_noise.npy'))
    train_ivar = np.load(os.path.join(dat_dir, 
        f'train.v0.1.{ibatch}.ivar.nde_noise.npy'))
    train_zred = np.load(os.path.join(dat_dir, 
        f'train.v0.1.{ibatch}.redshifts.npy'))

    # mask nonsense 
    train_mask = ~np.isfinite(train_seds)
    train_seds[train_mask] = 0.

    # get nomralization 
    train_norm = np.nanmedian(train_seds, axis=1)

    train_w = torch.tensor(train_ivar * ~train_mask * (train_norm**2)[:,None],
            dtype=torch.float32).to(device)
    train_y = torch.tensor(train_seds / train_norm[:, None],
            dtype=torch.float32).to(device)
    train_z = torch.tensor(train_zred, dtype=torch.float32).to(device)

    # load SDSS wavelength 
    wave_obs = torch.tensor(
            np.load('/scratch/network/chhahn/sedflow/sdss.clean.wave.npy'), 
            dtype=torch.float32)
    sdss = Instrument(wave_obs.to(device))

    # mask out emission lines 
    emline_mask_train = np.zeros(train_w.shape).astype(bool)

    w_lines = np.array([
        1033.3  , 1215.67 , 1239.42 , 1305.53 , 1335.52 , 1399.8  ,
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

    # load in best encoder model
    best_model = torch.load(os.path.join(
        '/scratch/network/chhahn/sedflow/encoder_spec/', 
        f"encoder_spec.emline_mask.{n_latent}latent.{i_model}.pt"), map_location=torch.device('cpu'))
    best_model.to(device)

    h_train = []
    for batch in train_loader:
        spec, w, z = batch
        h, _, _ = best_model._forward(spec, w, instrument=sdss, z=z)
        h_train.append(h.detach().cpu())
    h_train = np.concatenate(h_train, axis=0)
    
    np.save(os.path.join(dat_dir, f'train.v0.1.{ibatch}.norm_spec.nde_noise.npy'), 
            train_norm)
    np.save(os.path.join(dat_dir, f'train.v0.1.{ibatch}.h_spec.nde_noise.npy'), 
            h_train)
    return None 


if __name__=="__main__":
    task = sys.argv[1]
    if task == 'sample_nde_noise': 
        ibatch  = int(sys.argv[2])
        isplit  = int(sys.argv[3])
        arch    = sys.argv[4]
        sample_nde_noise(ibatch, isplit, arch)
    elif task == 'apply_nde_noise':
        ibatch   = int(sys.argv[2])
        n_latent = int(sys.argv[3])
        i_model  = int(sys.argv[4])
        apply_nde_noise(ibatch, n_latent, i_model)
    elif task == 'encode_spec': 
        ibatch   = int(sys.argv[2])
        n_latent = int(sys.argv[3])
        i_model  = int(sys.argv[4])
        encode_spec(ibatch, n_latent, i_model)
