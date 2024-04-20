'''

script to generate training SED data for the SEDflow project

'''
import os,sys
import psutil
import numpy as np
import multiprocessing as mp 
# --- provabgs --- 
from provabgs import infer as Infer
from provabgs import models as Models
# -- SEDflow --- 
from sedflow import util as U

####################################################
# input 
####################################################
Nsample = int(sys.argv[1])
seed    = int(sys.argv[2])
n_cpu   = psutil.cpu_count(logical=False)
dat_dir = os.path.join(U.data_dir(), 'seds')
####################################################
print('using %i CPUs to run %i SEDs' % (n_cpu, Nsample))

# SPS parameter priors 
prior_sps = Infer.load_priors([
    Infer.UniformPrior(6., 13., label='sed'),
    Infer.FlatDirichletPrior(4, label='sed'),           # flat dirichilet priors
    Infer.UniformPrior(0., 1., label='sed'),            # burst fraction
    Infer.UniformPrior(1e-2, 13.27, label='sed'),       # tburst
    Infer.LogUniformPrior(4.5e-5, 1.5e-2, label='sed'), # log uniform priors on ZH coeff
    Infer.LogUniformPrior(4.5e-5, 1.5e-2, label='sed'), # log uniform priors on ZH coeff
    Infer.UniformPrior(0., 3., label='sed'),            # uniform priors on dust1
    Infer.UniformPrior(0., 3., label='sed'),            # uniform priors on dust2
    Infer.UniformPrior(-2., 1., label='sed'),           # uniform priors on dust_index
    Infer.UniformPrior(0.1, 15., label='sed'),          # uniform priors on Umin based on Leja+(2017)
    Infer.UniformPrior(0.0, 0.15, label='sed'),         # uniform priors on gamma_e based on Leja+(2017)
    Infer.UniformPrior(0.1, 0.7, label='sed')           # uniform priors on Q_PAH based on Leja+(2017)
])

# SPS model 
m_sps = Models.NMF_NDE(burst=True, emulator=False)
m_sps._ssp_initiate()

# draw (parameters, z) from the prior
print('Drawing theta_SPS and redshifts') 
np.random.seed(seed)

thetas_sps  = np.array([prior_sps.transform(prior_sps.sample()) for i in range(Nsample)])
thetas_unt  = prior_sps.untransform(thetas_sps)
redshifts   = np.random.uniform(0, 1., Nsample) 
thetas      = np.concatenate([thetas_sps, np.atleast_2d(redshifts).T], axis=1)

####################################################
# generate SED for (parameters, z) values 
####################################################
print('Generating SED(theta_SPS, redshifts)') 
def SED(tt): 
    w, f = m_sps.sed(tt[:-1], tt[-1]) 
    return [w, f]

with mp.Pool(n_cpu) as p: 
    wfps = p.map(SED, thetas)

waves   = np.array([wfp[0] for wfp in wfps])
seds    = np.array([wfp[1] for wfp in wfps]) 

np.savez_compressed(os.path.join(dat_dir, 'modelb', 'train_sed.modelb.%i.thetas_sps.npz' % seed), thetas_sps)
np.savez_compressed(os.path.join(dat_dir, 'modelb', 'train_sed.modelb.%i.thetas_unt.npz' % seed), thetas_unt)
np.savez_compressed(os.path.join(dat_dir, 'modelb', 'train_sed.modelb.%i.redshifts.npz' % seed), redshifts)
np.savez_compressed(os.path.join(dat_dir, 'modelb', 'train_sed.modelb.%i.waves.npz' % seed), waves)
np.savez_compressed(os.path.join(dat_dir, 'modelb', 'train_sed.modelb.%i.seds.npz' % seed), seds)
