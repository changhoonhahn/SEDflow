'''

script to generate training Msurv data for the SEDflow project. This is used to
convert Mform to Mstar 

'''
import os,sys
import psutil
import numpy as np
import multiprocessing as mp 
# --- provabgs --- 
from provabgs import models as Models
# -- SEDflow --- 
import util as U

####################################################
# input 
####################################################
seed    = int(sys.argv[1])
n_cpu   = psutil.cpu_count(logical=False)
dat_dir = os.path.join(U.data_dir(), 'seds')
####################################################
print('using %i CPUs to run Msurvs' % (n_cpu))

# SPS model 
m_sps = Models.NMF_NDE(burst=True, emulator=False)
m_sps._ssp_initiate()

# draw (parameters, z) from the prior
print('read theta_SPS and sample t_age') 
thetas_sps  = np.load(os.path.join(dat_dir, 'modelb', 'train_sed.modelb.%i.thetas_sps.npz' % seed))['arr_0'] 
t_age       = np.random.uniform(0., 13.7, size=thetas_sps.shape[0]) 

thetas_sps[:,0] = 0.
thetas      = np.concatenate([thetas_sps, t_age[:,None]], axis=1)
####################################################
# generate Msurv for (parameters, tage) values 
####################################################
print('Generating Msurv(theta_SPS, tage)') 
def Msurv(tt): 
    #msurv = m_sps._surviving_mass(tt[:-1], tt[-1], emulator=False) 
    msurv_nmf = m_sps._surviving_mass_nmf(tt[:-1], tt[-1], emulator=False) 
    msurv_burst = m_sps._surviving_mass_burst(tt[:-1], tt[-1], emulator=False) 

    return [msurv_nmf, msurv_burst]

with mp.Pool(n_cpu) as p: 
    wfps = p.map(Msurv, thetas)

fsurvs_nmf = np.array([wfp[0] for wfp in wfps])
fsurvs_burst = np.array([wfp[1] for wfp in wfps])

np.save(os.path.join(dat_dir, 'modelb', 'train_sed.modelb.%i.msurv_t_age.npz' % seed), t_age)
np.save(os.path.join(dat_dir, 'modelb', 'train_sed.modelb.%i.msurv_fsurv_nmf.npz' % seed),
        fsurvs_nmf)
np.save(os.path.join(dat_dir, 'modelb', 'train_sed.modelb.%i.msurv_fsurv_burst.npz' % seed), 
        fsurvs_burst)
