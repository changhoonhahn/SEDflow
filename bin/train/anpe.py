import os, sys
import numpy as np
import pickle 
from sedflow import train as Train
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from sbi import utils as Ut
from sbi import inference as Inference


sample = sys.argv[1]
itrain = int(sys.argv[2]) 
nhidden = int(sys.argv[3]) 
nblocks = int(sys.argv[4])
print('%s training data; model %i' % (sample, itrain))

if torch.cuda.is_available(): device = 'cuda'
else: device = 'cpu'

# load training data (theta_sps, mag, sig, z) 
x_train, y_train = Train.load_data('train', version=1, sample=sample, params='thetas_unt') 
# convert gamma1, gamma2 (ZH NMF coefficients) to log space 
x_train[:,6] = np.log10(x_train[:,6])
x_train[:,7] = np.log10(x_train[:,7])
print('Ntrain = %i' % x_train.shape[0])
print('%i dim theta; %i dim obs' % (x_train.shape[1], y_train.shape[1]))


# prior 
prior_low   = [7, 0., 0., 0., 0., 1e-2, np.log10(4.5e-5), np.log10(4.5e-5), 0, 0., -2.]
prior_high  = [12.5, 1., 1., 1., 1., 13.27, np.log10(1.5e-2), np.log10(1.5e-2), 3., 3., 1.]
lower_bounds = torch.tensor(prior_low).to(device)
upper_bounds = torch.tensor(prior_high).to(device)
prior = Ut.BoxUniform(low=lower_bounds, high=upper_bounds, device=device)


# train NPE  
fanpe = os.path.join(Train.data_dir(), 'anpe_thetaunt_magsigz.%s.%ix%i.%i.pt' % (sample, nhidden, nblocks, itrain))
fsumm = os.path.join(Train.data_dir(), 'anpe_thetaunt_magsigz.%s.%ix%i.%i.p' % (sample, nhidden, nblocks, itrain))

anpe = Inference.SNPE(
        prior=prior, 
        density_estimator=Ut.posterior_nn('maf', hidden_features=nhidden, num_transforms=nblocks), 
        device=device)
anpe.append_simulations(
    torch.as_tensor(x_train.astype(np.float32)), 
    torch.as_tensor(y_train.astype(np.float32)))
p_x_y_estimator = anpe.train()#dataloader_kwargs={'num_workers': 4})

# save trained ANPE
torch.save(p_x_y_estimator.state_dict(), fanpe)

# save training summary
print(anpe._summary['best_validation_log_probs'])
pickle.dump(anpe._summary, open(fsumm, 'wb'))
