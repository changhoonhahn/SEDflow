'''

module for galaxy SED models

'''
import os 
import numpy as np 
import scipy.interpolate as Interp
from astropy.cosmology import Planck13

import torch

from . import util as U

class galaxySED(object): 
    ''' class object for galaxy SED model.
    '''
    def __init__(self, name=None, device=None):
        if name not in ['modelb.lowz', 'modelb.highz']: 
            raise NotImplementedError("currently only Model B for 0 < z < 1 and 1 < z < 2 implemented") 
        self._name = name 
        if device is None: 
            self.device = torch.device('cpu') 
        else: 
            self.device = device
    

class ModelB(galaxySED): 
    ''' class object for galaxy SED model. Currently the following models are supported: 
    
    * `modelb.lowz`: standard `provabgs` setup plus nebular emission and dust emission,
        which adds 3 additional parameters. Only supports redshift range: 0 < z < 1. The 
        parameters are: Mform, beta1, beta2, beta3, beta4, f_burst, t_burst, gamma1, gamma2, 
        tau_bc, tau_ism, dust_index, Umin, gamma_e, Q_PAH with the following prior: 

        UniformPrior(6., 13., label='sed'),
        FlatDirichletPrior(4, label='sed'),           # flat dirichilet priors
        UniformPrior(0., 1., label='sed'),            # burst fraction
        UniformPrior(1e-2, 13.27, label='sed'),       # tburst
        LogUniformPrior(4.5e-5, 1.5e-2, label='sed'), # log uniform priors on ZH coeff
        LogUniformPrior(4.5e-5, 1.5e-2, label='sed'), # log uniform priors on ZH coeff
        UniformPrior(0., 3., label='sed'),            # uniform priors on dust1
        UniformPrior(0., 3., label='sed'),            # uniform priors on dust2
        UniformPrior(-2., 1., label='sed'),           # uniform priors on dust_index
        UniformPrior(0.1, 15., label='sed'),          # uniform priors on Umin based on Leja+(2017)
        UniformPrior(0.0, 0.15, label='sed'),         # uniform priors on gamma_e based on Leja+(2017)
        UniformPrior(0.1, 0.7, label='sed')           # uniform priors on Q_PAH based on Leja+(2017)
    ])

    * `modelb.highz`: standard `provabgs` setup plus nebular emission and dust emission,
        which adds 3 additional parameters. Only supports redshift range: 1 < z < 2. The 
        parameters are: Mform, beta1, beta2, beta3, beta4, f_burst, t_burst, gamma1, gamma2, 
        tau_bc, tau_ism, dust_index, Umin, gamma_e, Q_PAH. Same prior as above.


    '''
    def __init__(self, name='modelb.lowz', device=None):
        if name not in ['modelb.lowz', 'modelb.highz']: 
            raise NotImplementedError("currently only Model B for 0 < z < 1 and 1 < z < 2 implemented") 
        super().__init__(name=name, device=device) 
        
        self._msurv_nmf_emu = None 
        self._msurv_burst_emu = None

        self._prior_type = 'uniform'

        self.cosmo = Planck13

        self._Z_min = 4.49043431e-05
        self._Z_max = 4.49043431e-02

        self._load_nmf_bases()

    def _redshift_check(self, zred): 
        ''' check that zred is valid for given galaxy SED model 
        '''
        if self._name == 'modelb.lowz': 
            # only 0 < z < 1 is supported for this model
            return zred > 0. and zred < 1.
        elif self._name == 'modelb.highz': 
            return zred > 1. and zred < 2.

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
        if self._name not in ['modelb.lowz', 'modelb.highz']:
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
            os.path.dirname(os.path.realpath(__file__)), 'dat', 'modelb', 'theta_shift.npy'))
        self._msurv_theta_scale = np.load(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'dat', 'modelb', 'theta_scale.npy'))
        
        self._msurv_nmf_shift = np.load(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'dat', 'modelb', 'msurv_nmf_shift.npy'))
        self._msurv_nmf_scale = np.load(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'dat', 'modelb', 'msurv_nmf_scale.npy'))
        
        self._msurv_burst_shift = np.load(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'dat', 'modelb', 'msurv_burst_shift.npy'))
        self._msurv_burst_scale = np.load(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'dat', 'modelb', 'msurv_burst_scale.npy'))
        
        # load Msurv emulator for NMF
        self._msurv_nmf_emu = torch.load(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'dat', 'modelb', 'emu_msurv.v2.nmf.pt'), 
            map_location=self.device)
        self._msurv_nmf_emu.to(self.device)

        # load Msurv emulator for burst 
        self._msurv_burst_emu = torch.load(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'dat', 'modelb', 'emu_msurv.v2.burst.pt'), 
            map_location=self.device)
        self._msurv_burst_emu.to(self.device)
        return None

    def SFH(self, tt, zred): 
        ''' star formation history for given set of parameter values and redshift. 
        SFH is parameterized using a smooth NMF component plus a star burst component.
    
        Parameters
        ----------
        tt : 1d or 2d array 
            Nparam or NxNparam array of parameter values 
        tage : float
            age of the galaxy 

        Returns
        -------
        tedges: 1d array 
            bin edges of log-spaced look back time
        sfh : 1d or 2d array 
            star formation history in Msun/yr
        '''
        theta = np.atleast_2d(tt).copy() 
        logm = theta[:,0]
        theta_sfh_nmf = theta[:,1:5] # sfh nmf basis coefficients 
        fburst = theta[:,5] # burst fraction 
        tburst = theta[:,6] # burst time 

        tage = self.cosmo.age(zred).value # age in Gyr
        tlb_edges = self._tlookback_bin_edges(tage) # log-spaced lookback time bin edges 
    
        # nmf 
        sfh_basis_tlb = np.array([
            U.trapz_rebin(self._t_lb_hr, _sfh_basis, edges=tlb_edges) 
            for _sfh_basis in self._sfh_basis_hr])

        sfh = np.sum(np.array([theta_sfh_nmf[:,i][:,None] *
            sfh_basis_tlb[i][None,:] for i in range(self._Ncomp_sfh)]),
            axis=0)

        sfh /= np.sum(1e9 * np.diff(tlb_edges) * sfh, axis=1)[:,None] # normalize 
      
        # starburst 
        noburst = (tburst > tage)
        fburst[noburst] = 0. 

        sfh *= (1. - fburst) #  normalized nmf componenet 
    
        # calculate star burst component 
        dts = np.diff(tlb_edges)
        
        iburst = np.digitize(tburst[~noburst], tlb_edges)-1 # bin with burst

        sfh_burst = np.zeros(sfh.shape)
        sfh_burst[~noburst, iburst] += 1. / (1e9 * dts[iburst])

        sfh += fburst * sfh_burst # add starburst 

        # multiply by stellar mass 
        sfh *= 10**logm

        return tlb_edges, sfh 
    
    def avgSFR(self, tt, zred, dt=1):
        ''' calculate SFR averaged over `dt` Gyr 

        parameters
        ----------
        tt : array_like[Ntheta, Nparam]
            parameters of SED model 
        zred : float 
            redshift
        dt : float
            Gyrs to average the SFHs 

        returns
        -------
        avgsfr : 
            average SFR in Msun/yr
        '''
        theta = np.atleast_2d(tt).copy() 
        logm = theta[:,0]
        theta_sfh_nmf = theta[:,1:5] # sfh nmf basis coefficients 
        fburst = theta[:,5] # burst fraction 
        tburst = theta[:,6] # burst time 

        tage = self.cosmo.age(zred).value # age in Gyr
        tlb_edges = self._tlookback_bin_edges(tage) # log-spaced lookback time bin edges 
    
        # nmf 
        sfh_basis_tlb = np.array([
            U.trapz_rebin(self._t_lb_hr, _sfh_basis, edges=tlb_edges) 
            for _sfh_basis in self._sfh_basis_hr])

        sfh = np.sum(np.array([theta_sfh_nmf[:,i][:,None] *
            sfh_basis_tlb[i][None,:] for i in range(self._Ncomp_sfh)]),
            axis=0)

        sfh /= np.sum(np.diff(tlb_edges) * sfh, axis=1)[:,None] # normalize 
    
        # calculate stellar mass formed over 0-dt 
        i_dt = np.digitize(dt, tlb_edges) - 1
        Mform = np.sum(np.diff(tlb_edges)[:i_dt][None,:] * sfh[:,:i_dt], axis=1) 
        Mform += (dt - tlb_edges[i_dt]) * sfh[:,i_dt]
        
        # add starburst 
        noburst = (tburst > tage)
        fburst[noburst] = 0. 
        Mform *= (1. - fburst)
        
        burst_dt = (tburst < dt)
        Mform[burst_dt] += fburst[burst_dt]

        # multiply by total mass  
        return Mform *  10**logm  / dt / 1e9

    def ZH(self, tt, zred): 
        ''' metallicity history for given set of parameters. metallicity is
        parameterized using a 2 component NMF basis. The parameter values
        specify the coefficient of the components and this method returns the
        linear combination of the two. 

        parameters
        ----------
        tt : array_like[N,Nparam]
           parameter values of the model 

        zred : float
            redshift of galaxy/csp

        Returns 
        -------
        tedge : 1d array
            bin edges of lookback time

        zh : 2d array
            metallicity at cosmic time t --- ZH(t) 
        '''
        tage = self.cosmo.age(zred).value # age in Gyr
    
        theta = np.atleast_2d(tt).copy() 
        theta_zh = theta[:,7:9] # metallicity history basis coefficients  

        # log-spaced lookback time bin edges 
        tlb_edges = self._tlookback_bin_edges(tage)
        tlb = 0.5 * (tlb_edges[1:] + tlb_edges[:-1])

        # metallicity basis
        _z_basis = np.array([self._zh_basis[i](tlb) for i in range(self._Ncomp_zh)]) 

        # get metallicity history
        zh = np.sum(np.array([theta_zh[:,i][:,None] * _z_basis[i][None,:] for i in
            range(self._Ncomp_zh)]), axis=0).clip(self._Z_min, self._Z_max) 

        return tlb_edges, zh 
    
    def Z_MW(self, tt, zred):
        ''' calculate mass weighted metallicity of the galaxy . 
        '''
        theta = np.atleast_2d(tt).copy()
        logm = theta[:,0]

        tlb_edge, sfh = self.SFH(tt, zred) # SFH 
        _, zh = self.ZH(tt, zred)  # ZH 

        # mass weighted average
        z_mw = np.sum(1e9 * np.diff(tlb_edge)[None,:] * sfh * zh, axis=1) / (10**logm) 
        return z_mw 

    def tage_MW(self, tt, zred):
        ''' calculate mass weighted age of the galaxy 
        '''
        theta = np.atleast_2d(tt).copy()
        logm = theta[:,0]

        tlb_edge, sfh = self.SFH(tt, zred) # get SFH 
        th = 0.5 * (tlb_edge[1:] + tlb_edge[:-1]) 

        # mass weighted average
        t_mw = np.sum(1e9 * np.diff(tlb_edge)[None,:] * sfh * th, axis=1) / (10**logm) 
        return t_mw 

    def _tlookback_bin_edges(self, tage): 
        ''' hardcoded log-spaced lookback time bin edges. Bins have 0.1 log10(Gyr)
        widths. See `nb/tlookback_binning.ipynb` for comparison of linear and
        log-spaced binning. With log-space we reproduce spectra constructed from
        high time resolution SFHs more accurately with fewer stellar populations.
        '''
        bin_edges = np.zeros(43)
        bin_edges[1:-1] = 10**(6.05 + 0.1 * np.arange(41) - 9.)
        bin_edges[-1] = 13.8
        if tage is None: 
            return bin_edges
        else: 
            return np.concatenate([bin_edges[bin_edges < tage], [tage]])

    def _load_nmf_bases(self):
        ''' load non-negative matrix factorization basese used for the Model B SFH and ZH 
        '''
        dir_dat = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dat', 'modelb') 

        fsfh = os.path.join(dir_dat, 'NMF_2basis_SFH_components_nowgt_lin_Nc4.txt')
        fzh = os.path.join(dir_dat, 'NMF_2basis_Z_components_nowgt_lin_Nc2.txt') 
        ft = os.path.join(dir_dat, 'sfh_t_int.txt') 

        nmf_sfh = np.loadtxt(fsfh)[:,::-1] # basis order is jumbled up it should be [2 ,0, 1, 3]
        nmf_zh  = np.loadtxt(fzh)[:,::-1] 
        nmf_t   = np.loadtxt(ft)[::-1] # look back time 

        self._nmf_t_lb_sfh      = nmf_t 
        self._nmf_t_lb_zh       = nmf_t 
        self._nmf_sfh_basis     = np.array([nmf_sfh[2], nmf_sfh[0], nmf_sfh[1], nmf_sfh[3]])
        self._nmf_zh_basis      = nmf_zh

        self._Ncomp_sfh = self._nmf_sfh_basis.shape[0]
        self._Ncomp_zh = self._nmf_zh_basis.shape[0]
        
        # SFH bases as a function of lookback time 
        self._sfh_basis = [
                Interp.InterpolatedUnivariateSpline(
                    self._nmf_t_lb_sfh, 
                    self._nmf_sfh_basis[i], k=1) 
                for i in range(self._Ncomp_sfh)
                ]
        self._zh_basis = [
                Interp.InterpolatedUnivariateSpline(
                    self._nmf_t_lb_zh, 
                    self._nmf_zh_basis[i], k=1) 
                for i in range(self._Ncomp_zh)]
        
        # high resolution tabulated SFHs used in the SFH calculation
        self._t_lb_hr       = np.linspace(0., 13.8, int(5e4))
        self._sfh_basis_hr  = [sfh_basis(self._t_lb_hr) for sfh_basis in self._sfh_basis]
        return None 
