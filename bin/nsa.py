'''

script to apply trained ANPE on the NSA data 

'''
import os, sys
import numpy as np
from scipy import stats
from sedflow import obs as Obs
from sedflow import train as Train

# torch
import torch
from sbi import utils as Ut
from sbi import inference as Inference


sample = sys.argv[1]
itrain = int(sys.argv[2]) 
nhidden = int(sys.argv[3]) 
nblocks = int(sys.argv[4])
print('%s training data; model %i' % (sample, itrain))

ichunk  = int(sys.argv[5])
assert ichunk < 34

####################################################################
# load NSA observations 
# y = [u, g, r, i, z, sigma_u, sigma_g, sigma_r, sigma_i, sigma_z, z]
####################################################################
y_nsa = Obs.load_nsa_data(test_set=False)

####################################################################
# set prior (this is fixed) 
####################################################################
prior_low = [7, 0., 0., 0., 0., 1e-2, np.log10(4.5e-5), np.log10(4.5e-5), 0, 0., -2.]
prior_high = [12.5, 1., 1., 1., 1., 13.27, np.log10(1.5e-2), np.log10(1.5e-2), 3., 3., 1.]
lower_bounds = torch.tensor(prior_low)
upper_bounds = torch.tensor(prior_high)
prior = Ut.BoxUniform(low=lower_bounds, high=upper_bounds, device='cpu')

####################################################################
# load trained ANPE 
####################################################################
# load test data (solely used for ANPE initialization)
x_test, y_test = Train.load_data('test', version=1, sample=sample, params='thetas_unt')
x_test[:,6] = np.log10(x_test[:,6])
x_test[:,7] = np.log10(x_test[:,7])

fanpe = os.path.join(Train.data_dir(), 'anpe_thetaunt_magsigz.%s.%ix%i.%i.pt' % (sample, nhidden, nblocks, itrain))

anpe = Inference.SNPE(prior=prior, 
                      density_estimator=Ut.posterior_nn('maf', hidden_features=nhidden, num_transforms=nblocks), 
                      device='cpu')
anpe.append_simulations(
    torch.as_tensor(x_test.astype(np.float32)), 
    torch.as_tensor(y_test.astype(np.float32)))

p_x_y_estimator = anpe._build_neural_net(torch.as_tensor(x_test.astype(np.float32)), torch.as_tensor(y_test.astype(np.float32)))
p_x_y_estimator.load_state_dict(torch.load(fanpe))

anpe._x_shape = Ut.x_shape_from_simulation(torch.as_tensor(y_test.astype(np.float32)))

hatp_x_y = anpe.build_posterior(p_x_y_estimator)

####################################################################
# estimate posteriors  
####################################################################

def get_posterior(y_nsa_i, nmcmc=10000): 
    ''' given [mag, uncertainty, redshift] of a galaxy, draw nmcmc samples of
    the posterior. 
    '''
    mcmc_anpe = hatp_x_y.sample((nmcmc,), x=torch.as_tensor(y_nsa_i), 
            show_progress_bars=False)
    return np.array(mcmc_anpe) 

mcmcs = [] 
for igal in np.arange(y_nsa.shape[0])[ichunk*1000:(ichunk+1)*1000]: 
    _mcmc_i = get_posterior(y_nsa[igal])
    mcmcs.append(_mcmc_i)

# save samples 
np.save(fanpe.replace('.pt', '.nsa%iof34.samples.npy' % ichunk),
        np.array(mcmcs))
