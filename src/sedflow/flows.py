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


def read_EnsembleFlow(study_name, n_ensemble=5, device='cpu', name='modela'): 
    ''' read in ensemble of best flows
    '''
    fqphis = U.read_best_fqphi(study_name, n_ensemble=n_ensemble, name=name)

    flos = []     
    for fqphi in fqphis: 
        _flo = Flow(device=device) 
        _flo.load_flow(fqphi)

        flos.append(_flo)
    return flos 

