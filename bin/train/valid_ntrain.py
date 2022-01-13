'''

script to validate trained ANPE

'''
import os, sys
import numpy as np
from scipy import stats
from sedflow import train as Train

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from sbi import utils as Ut
from sbi import inference as Inference

# --- plotting ---
import corner as DFM
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.xmargin'] = 1
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.labelsize'] = 'x-large'
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['legend.frameon'] = False

sample = sys.argv[1]
itrain = int(sys.argv[2]) 
nhidden = int(sys.argv[3]) 
nblocks = int(sys.argv[4])
ntrain = int(sys.argv[5])
print('%s training data; model %i' % (sample, itrain))

# load test data ---  x = theta_sps, y = [u, g, r, i, z, sigma_u, sigma_g, sigma_r, sigma_i, sigma_z, z]
x_test, y_test = Train.load_data('test', version=1, sample=sample, params='thetas_unt')
x_test[:,6] = np.log10(x_test[:,6])
x_test[:,7] = np.log10(x_test[:,7])

# prior 
prior_low = [7, 0., 0., 0., 0., 1e-2, np.log10(4.5e-5), np.log10(4.5e-5), 0, 0., -2.]
prior_high = [12.5, 1., 1., 1., 1., 13.27, np.log10(1.5e-2), np.log10(1.5e-2), 3., 3., 1.]
lower_bounds = torch.tensor(prior_low)
upper_bounds = torch.tensor(prior_high)
prior = Ut.BoxUniform(low=lower_bounds, high=upper_bounds, device='cpu')


def pps(hat_p_x_y, ntest=100, nmcmc=10000):
    ''' given npe, calculate pp for ntest test data
    '''
    mcmcs = [] 
    pp_thetas, rank_thetas = [], []
    for igal in np.arange(ntest):
        _mcmc_anpe = hatp_x_y.sample((nmcmc,), x=torch.as_tensor(y_test[igal]), show_progress_bars=False)

        pp_theta, rank_theta = [], []
        for itheta in range(_mcmc_anpe.shape[1]):
            pp_theta.append(stats.percentileofscore(_mcmc_anpe[:,itheta], x_test[igal,itheta])/100.)
            rank_theta.append(np.sum(np.array(_mcmc_anpe[:,itheta]) < x_test[igal,itheta]))
        pp_thetas.append(pp_theta)
        rank_thetas.append(rank_theta)
        mcmcs.append(np.array(_mcmc_anpe))

    pp_thetas = np.array(pp_thetas)
    rank_thetas = np.array(rank_thetas)
    mcmcs = np.array(mcmcs)
    return pp_thetas, rank_thetas, mcmcs


# load ANPE 
fanpe = os.path.join(Train.data_dir(), 'anpe_thetaunt_magsigz.%s.ntrain%i.%ix%i.%i.pt' % (sample, ntrain, nhidden, nblocks, itrain))

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

# calculate percentile scores and rank 
_pp, _rank, _mcmcs = pps(hatp_x_y, ntest=1000, nmcmc=10000)

# save samples 
np.save(fanpe.replace('.pt', '.samples.npy'), _mcmcs)

ks_p = []
for ii in range(_pp.shape[1]): 
    _, _ks_p = stats.kstest(_pp[ii], 'uniform')
    ks_p.append(_ks_p)

_, _ks_tot_p = stats.kstest(_pp.flatten(), 'uniform')
print(fanpe, _ks_tot_p)

theta_lbls = [r'$\log M_*$', r"$\beta'_1$", r"$\beta'_2$", r"$\beta'_3$", r'$f_{\rm burst}$', r'$t_{\rm burst}$', r'$\log \gamma_1$', r'$\log \gamma_2$', r'$\tau_1$', r'$\tau_2$', r'$n_{\rm dust}$']
fig = plt.figure(figsize=(8,8))
sub = fig.add_subplot(111)
for itheta in range(_pp.shape[1]): 
    # evaluate the histogram
    values, base = np.histogram(_pp[:,itheta], bins=40)
    #evaluate the cumulative
    cumulative = np.cumsum(values) / np.sum(values)
    sub.plot(base[:-1], cumulative, label=theta_lbls[itheta])
sub.plot([0., 1.], [0., 1.], c='k', ls='--')
sub.set_xlim(0., 1.)
sub.set_ylim(0., 1.)
sub.set_title('%.5e' % _ks_tot_p, fontsize=25)
sub.legend(loc='upper left')
fig.savefig(fanpe.replace('.pt', '.ppplot.png'))
