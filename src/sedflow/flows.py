import os 
import glob
import numpy as np 

import torch
from torch import nn 

from sbi import utils as Ut
from sbi import inference as Inference


from . import util as U


class Flow(object): 
    ''' Load/train/evaluate flows for SEDflow. This is basically a wrapper for `sbi` tailored for SED modeling 
    to make it easier to use. It is recommended 
    '''
    
    def __init__(self, device=None): 
        if device is None: 
            self.device = torch.device('cpu') 
        else: 
            self.device = device

        self.prior = None 

        self.flow = None 

    def load_flow(self, fqphi): 
        ''' load flow from saved 

        parameters
        ----------
        fqphi : str
            name of torch save file. 
        '''
        self.flow = torch.load(fqphi, map_location=self.device)
        self.prior_type = self.flow._prior_type
        self.prior_low = self.flow._prior_low
        self.prior_high = self.flow._prior_high
        return None 

    def sample(self, sed, Nsample=10000, progress_bar=False): 
        ''' sample flow at given SED photometry --- theta' ~ q_phi(theta | X_obs) 

        parameters
        ----------
        sed : array_like
            SED photometry
        
        Nsample : int
            number of samples to draw 

        progress_bar : bool
            toggle progress bar 


        '''
        _thetas = self.flow.sample((Nsample,), x=torch.as_tensor(sed).to(self.device), 
                show_progress_bars=progress_bar)
        _thetas = _thetas.detach().cpu().numpy()

        if self.flow._prior_type == 'uniform':
            # transform back
            thetas = np.empty(_thetas.shape)
            for i in range(_thetas.shape[1]): 
                thetas[:,i] = U.cdf_transform(_thetas[:,i], [self.prior_low[i], self.prior_high[i]])
            return thetas
        else: 
            return _thetas 

    def train_flow(self, theta_train, sed_train, nde=None, summary_writer=None, training_batch_size=50, learning_rate=5e-4): 
        ''' train self.flow given training dataset of {(theta, SED)}

        parameters
        ----------
        theta_train : array like (N_train, N_theta)
            SED model parameters

        sed_train : array like (N_train, N_sed)
            SED photometry predited by model. 

        prior : prior object
            `sbi` prior object  e.g. sbi.utils.BoxUniform

        nde : posterior_nn object
            sbi.utils.posterior_nn object

        summary_writer : 
            summary writer for posterity: torch.utils.tensorboard.writer.SummaryWriter
        '''
        if self.prior is None: raise ValueError("set prior before training") 

        self.anpe = Inference.SNPE(
                    prior=self.prior_tf,
                    density_estimator=nde,
                    device=self.device, 
                    summary_writer=summary_writer) 
    
        # if prior are uniform distributions perform inverse CDF transform. This is to help with the hard border edges. 
        # See Li et al (2024) POPSED paper.
        if self.prior_type == 'uniform': 
            # check that the thetas are within the prior range; otherwise this will give nonsense
            assert np.all(np.min(theta_train, axis=0) - np.array(self.prior_low) > 0), 'theta is outside of prior range' 
            assert np.all(np.max(theta_train, axis=0) - np.array(self.prior_high) < 0), 'theta is outside of prior range' 

            _theta_train = np.empty(theta_train.shape)
            for i in range(theta_train.shape[1]): 
                _theta_train[:,i] = U.inv_cdf_transform(theta_train[:,i], [self.prior_low[i], self.prior_high[i]])
        else: 
            _theta_train = theta_train.copy()

        self.anpe.append_simulations(
            torch.as_tensor(_theta_train.astype(np.float32)).to(self.device),
            torch.as_tensor(sed_train.astype(np.float32)).to(self.device))

        p_x_y_est = self.anpe.train(
                training_batch_size=training_batch_size,
                learning_rate=learning_rate,  
                show_train_summary=True)

        # save trained NPE  
        self.flow = self.anpe.build_posterior(p_x_y_est)
        if self.prior_type == 'uniform': 
            self.flow._prior_type = 'uniform'
            self.flow._prior_low = self.prior_low
            self.flow._prior_high = self.prior_high

        return None
    
    def nde(self, architecture='maf', n_hidden=None, n_transf=None, n_blocks=None, p_drop=None, use_batch_norm=True):
        ''' define an NDE with a specific architecture 

        parameters 
        ----------
        architecture : str
            string specifying the architecture 

        n_hidden : int
            number of hidden features in the NN

        n_transf : int
            number of transformations

        n_blocks : int
            number of blocks

        p_drop : float
            dropout probability 

        return 
        ------
        nde : posterior_nn object
        '''
        nde = Ut.posterior_nn(architecture, 
                hidden_features=n_hidden, 
                num_transforms=n_transf, 
                num_blocks=n_blocks, 
                dropout_probability=p_drop, 
                use_batch_norm=use_batch_norm)
        return nde 

    def set_prior(self, prior_type, **kwargs): 
        ''' set prior 
        '''
        self.prior_type = 'uniform' 

        if prior_type == 'uniform': 
            # simple uniform prior based on lower bound and upper bound 
            low, high = kwargs['low'], kwargs['high']
            self.prior_low  = low
            self.prior_high = high

            lower_bounds = torch.tensor(low).to(self.device)
            upper_bounds = torch.tensor(high).to(self.device)

            self.prior = Ut.BoxUniform(low=lower_bounds, high=upper_bounds, device=self.device)
            
            # transformed prior
            self.prior_tf = Ut.BoxUniform(
                    low=torch.tensor([-10. for i in range(len(low))]), 
                    high=torch.tensor([10. for i in range(len(high))]), 
                    device=self.device)

        else: 
            raise NotImplementedError 

        return self.prior


class DESIflow(Flow): 
    ''' SEDflow specifically designed for DESI. It assumes the inputs are grzW1W2 photometry 
    and spectroscopic redshift

    DESIflow curentlys upports the following models: 
    * `modelb.lowz.grzW1W2`: standard `provabgs` setup plus nebular emission and dust emission,
        which adds 3 additional parameters. Only supports redshift range: 0 < z < 1. The 
        parameters are: Mform, beta1, beta2, beta3, beta4, f_burst, t_burst, gamma1, gamma2, 
        tau_bc, tau_ism, dust_index, Umin, gamma_e, Q_PAH. 

    '''
    def __init__(self, name='modelb.lowz.grzW1W2', device=None): 
        ''' load ensemble of SEDflows, specificially designed for analyzing DESI photometry.  

        parameters
        ----------
        name : str
            Name of DESI flow set up. Currently only Model B Low z (0 < z < 1) implemented. 

        device : str
            specify 'cpu' or 'cuda' if using gpu

        notes
        -----
        * currently only Model B low 0<z<1 implemented. 
        '''
        if name not in ['modelb.lowz.grzW1W2']: 
            raise NotImplementedError("currently only Model B low 0 < z < 1 implemented") 
        # model name that specifies the SED model, the redshift range, and the
        # photometric bands. 
        self._name = name 
        
        super().__init__(device=device) 

        self._load_desi_flow()
        
        self._msurv_nmf_emu = None 
        self._msurv_burst_emu = None

    def run(self, nmgy, sig_nmgy, zred, Nsample=10000, progress_bar=False): 
        ''' run DESIflow on specified photometry, photometric noise, and redshift 

        parameters
        ----------
        nmgy : array_like
            photometric flux in nanomaggies
        sig_nmgy : array_like 
            photometric noise in nanomaggies 
        zred : float
            redshift 

        returns
        -------
        thetas : array_like 
            Nsample x Ntheta array of samples drawn from the posterior 
    
        notes
        -----

        '''
        if self._name == 'modelb.lowz.grzW1W2':
            #this model is trained to take log10(flux), sig_flux, zred as input 
            assert len(nmgy) == 5, 'only grzW1W2 band photometry is supported for this model'
            assert len(sig_nmgy) == 5, 'only grzW1W2 band photometry is supported for this model'
            assert zred > 0., 'only 0 < z < 1 is supported for this model' 
            assert zred < 1., 'only 0 < z < 1 is supported for this model' 
        
            x_photo = np.concatenate([np.log10(nmgy), sig_nmgy, [zred]]) 
        else: 
            raise NotImplementedError

        _thetas = self.flow.sample((Nsample,), x=torch.as_tensor(x_photo).to(self.device), 
                show_progress_bars=progress_bar)
        _thetas = _thetas.detach().cpu().numpy()

        if self.flow._prior_type == 'uniform':
            # for uniform priors, SEDflow implements a default inverse CDF
            # transform. This transforms it back. 
            thetas = np.empty(_thetas.shape)
            for i in range(_thetas.shape[1]): 
                thetas[:,i] = U.cdf_transform(_thetas[:,i], [self.prior_low[i], self.prior_high[i]])
        else: 
            thetas = _thetas

        if self._name == 'modelb.lowz.grzW1W2':
            # Model B uses provabgs SED model, which requires the SFH NMF basis
            # coefficients to add up to one. SEDflow is trained in a
            # transformed space based on Betnacourt (2010): https://arxiv.org/abs/1010.3436
            # Here we transform it back to the original provabgs SED model
            # parameter space. 
            thetas_sfh = self._provabgs_sfh_dirichlet_transform(thetas[:,1:4])

            thetas = np.concatenate([thetas[:,0][:,None], thetas_sfh, thetas[:,4:]], axis=1) 
        else: 
            raise NotImplementedError

        return thetas

    def _load_desi_flow(self): 
        ''' load flow trained specifically for DESI 
        '''
        fqphis = glob.glob(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dat', self._name, 'qphi*'))
        if len(fqphis) != 1: 
            raise ValueError('currently only supports a single flow')

        _ = self.load_flow(fqphis[0]) 
        return None 

    def _provabgs_sfh_dirichlet_transform(self, tt): 
        ''' For models where we use the provabgs SED model, we transform the
        four SFH NMF basis coefficients, which are sampled using a Dirichlet
        distribuiton, to three values that are sampled from a uniform distribution.
        We use a warped manifold transform as specified in Betnacourt (2010): 
        https://arxiv.org/abs/1010.3436. This function transforms them back to 
        the original SFH NMF basis coefficients. 
        '''
        assert tt.shape[-1] == 3
        tt_d = np.empty(tt.shape[:-1]+(4,)) 
    
        tt_d[...,0] = 1. - tt[...,0]
        for i in range(1,3): 
            tt_d[...,i] = np.prod(tt[...,:i], axis=-1) * (1. - tt[...,i]) 
        tt_d[...,-1] = np.prod(tt, axis=-1) 
        return tt_d 

    def _provabgs_sfh_dirichlet_transform_inverse(self, tt): 
        ''' For models where we use the provabgs SED model, we transform the
        four SFH NMF basis coefficients, which are sampled using a Dirichlet
        distribuiton, to three values that are sampled from a uniform distribution.
        We use a warped manifold transform as specified in Betnacourt (2010): 
        https://arxiv.org/abs/1010.3436. This function transforms the original 
        SFH NMF basis coefficients to the uniform basis. 
        '''
        tt_u = np.zeros((tt.shape[0],3))
        tt_u[:,0] = (1. - tt[:,0]).clip(1e-8, None)
        for i in range(1,3):
            tt_u[:,i] = 1. - (tt[:,i] / np.prod(tt_u[:,:i], axis=1))
        return tt_u
    
    # galaxy properties 
    def _msurv(self, thetas, tage): 
        ''' calculate survivng mass fraction used to calculate M* from total
        formed mass

        parameters
        ----------
        thetas : [N, Ndim] array-like
            parameters for SED model. See class description for details.  
        tage : [N,] array-like
            age of the galaxy in Gyrs
        
        returns
        -------
        logmsurv : array-like
            log10 (surviving stellar mass)
        '''
        if self._name == 'modelb.lowz.grzW1W2':
            pass
        else: 
            raise NotImplementedError

        if self._msurv_nmf_emu is None or self._msurv_burst_emu is None: 
            self._load_msurv()

        thetas = np.atleast_2d(thetas).copy()
        tage = tage[:,None].copy()
    
        # nmf contribution 
        fsurv_nmf = self._msurv_nmf(thetas, tage)
    
        # burst contribution 
        fsurv_burst = self._msurv_burst(thetas, tage)
        # if tburst is older than the age of the galaxy. This should by construction 
        # never happen... 
        fsurv_burst[thetas[:,6] > tage[:,0]] = 0. 

        fsurv = (1. - thetas[:,5]) * fsurv_nmf.flatten() + thetas[:,5] * fsurv_burst.flatten()
        
        logmsurv = thetas[:,0] + np.log10(fsurv) 
        return logmsurv

    def _msurv_nmf(self, _thetas, tage): 
        ''' calculate survivng mass fraction used to calculate M* from total formed 
        mass for the NMF contribution 

        parameters
        ----------
        thetas : [N x Ndim] array
            Parameters of the SED model, provided in the same format as the output of the
            NPE.

        tage : [N x 1] array 
            age of the galaxy in Gyrs. 

        returns
        -------
        fsurv : array
            surviving mass fraction 
        '''
        # parse and transform input parameters  
        thetas = _thetas[:,1:9].copy() # extract SFH and ZH parameters only 
        
        # transform dirichlet prior back to uniform 
        thetas_sfh = self._provabgs_sfh_dirichlet_transform_inverse(thetas[:,:4])

        # gamma1, gamma2 to log10 
        thetas[:,6] = np.log10(thetas[:,6]) 
        thetas[:,7] = np.log10(thetas[:,7]) 
        
        thetas = np.concatenate([thetas_sfh, thetas[:,6:], tage], axis=1) 
        
        # whiten theta and select columns 
        thetas[:,:3] = (thetas[:,:3] - self._msurv_theta_shift[:3]) / self._msurv_theta_scale[:3]
        thetas[:,3:] = (thetas[:,3:] - self._msurv_theta_shift[5:]) / self._msurv_theta_scale[5:]
        thetas = torch.tensor(thetas.astype(np.float32)).to(self.device)

        # evaluate emulator 
        with torch.no_grad(): 
            _msurv = self._msurv_nmf_emu(thetas).cpu().numpy()

        return (_msurv * self._msurv_nmf_scale) + self._msurv_nmf_shift

    def _msurv_burst(self, thetas, tage): 
        ''' calculate survivng mass fraction used to calculate M* from total formed 
        mass for the burst contribution 

        parameters
        ----------
        thetas : [N, Ndim] array
            Parameters of the SED model, provided in the same format as the output of the
            NPE.

        tage : [N, 1] array 
            age of the galaxy in Gyrs. 

        returns
        -------
        fsurv : array
            surviving mass fraction 
        '''
        # parse and transform input parameters  
        thetas = thetas[:,1:9].copy() # extract SFH and ZH parameters only 
        
        # gamma1, gamma2 to log10 
        thetas[:,6] = np.log10(thetas[:,6]) 
        thetas[:,7] = np.log10(thetas[:,7]) 
        
        thetas = np.concatenate([thetas[:,5:], tage], axis=1) 
        
        # whiten theta and select columns 
        thetas = (thetas - self._msurv_theta_shift[4:]) / self._msurv_theta_scale[4:]
        thetas = torch.tensor(thetas.astype(np.float32)).to(self.device)

        # evaluate emulator 
        with torch.no_grad(): 
            _msurv = self._msurv_burst_emu(thetas).cpu().numpy()

        return (_msurv * self._msurv_burst_scale) + self._msurv_burst_shift

    def _load_msurv(self): 
        ''' load files necessary to run the surviving mass fraction emulator: 
        shift/scale arrays, emulator for NMF contribution, emulator for burst contribution. 
        '''
        # load shift/scale arrays
        self._msurv_theta_shift = np.load(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'dat', self._name, 'theta_shift.npy'))
        self._msurv_theta_scale = np.load(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'dat', self._name, 'theta_scale.npy'))
        
        self._msurv_nmf_shift = np.load(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'dat', self._name, 'msurv_nmf_shift.npy'))
        self._msurv_nmf_scale = np.load(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'dat', self._name, 'msurv_nmf_scale.npy'))
        
        self._msurv_burst_shift = np.load(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'dat', self._name, 'msurv_burst_shift.npy'))
        self._msurv_burst_scale = np.load(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'dat', self._name, 'msurv_burst_scale.npy'))
        
        # load Msurv emulator for NMF
        self._msurv_nmf_emu = torch.load(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'dat', self._name, 'emu_msurv.v2.nmf.pt'), 
            map_location=self.device)
        self._msurv_nmf_emu.to(self.device)

        # load Msurv emulator for burst 
        self._msurv_burst_emu = torch.load(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'dat', self._name, 'emu_msurv.v2.burst.pt'), 
            map_location=self.device)
        self._msurv_burst_emu.to(self.device)
        return None


class MLP(nn.Sequential):
    """Multi-Layer Perceptron
    A simple implementation with a configurable number of hidden layers and
    activation functions.
    Parameters
    ----------
    n_in: int
        Input dimension
    n_out: int
        Output dimension
    n_hidden: list of int
        Dimensions for every hidden layer
    act: list of callables
        Activation functions after every layer. Needs to have len(n_hidden) + 1
        If `None`, will be set to `LeakyReLU` for every layer.
    dropout: float
        Dropout probability
    """
    def __init__(self,
                 n_in,
                 n_out,
                 n_hidden=(16, 16, 16),
                 act=None,
                 dropout=0):

        if act is None:
            act = [ nn.LeakyReLU(), ] * (len(n_hidden) + 1)
        assert len(act) == len(n_hidden) + 1

        layer = []
        n_ = [n_in, *n_hidden, n_out]
        for i in range(len(n_)-1):
                layer.append(nn.Linear(n_[i], n_[i+1]))
                layer.append(act[i])
                layer.append(nn.Dropout(p=dropout))

        super(MLP, self).__init__(*layer)
