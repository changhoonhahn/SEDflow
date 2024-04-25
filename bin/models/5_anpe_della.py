'''


script to train Neural Posterior Estimators with Optuna hyperparameter optimization on della 
with the updated framework


'''
import os, sys 
import numpy as np

import torch
from torch import nn 
from torch.utils.tensorboard.writer import SummaryWriter

import optuna 

from sbi import utils as Ut
from sbi import inference as Inference

import util as U
from sedflow import flows as F

##################################################################################
# input 
##################################################################################
bands = sys.argv[1] # which bands passes
freez = (sys.argv[2] == True) 
model = sys.argv[3]
try: 
    study = sys.argv[4]
except: 
    study = ''
##################################################################################
cuda = torch.cuda.is_available()
device = ("cuda:0" if cuda else "cpu")

seed = 12387
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)
##################################################################################
# load data 
##################################################################################
if model == 'modela': 
    theta_train, X_train = U.load_modela('train', bands=bands, infer_redshift=freez)
elif model == 'modelb': 
    theta_train, X_train = U.load_modelb('train', bands=bands, infer_redshift=freez)
else:
    raise ValueError

print('Ntrain + Nvalid = %i' % (theta_train.shape[0]))

##################################################################################
# set prior range
##################################################################################
if model == 'modela': 
    if not freez: 
        prior_low   = [7, 0., 0., 0., 0., 1e-2, np.log10(4.5e-5), np.log10(4.5e-5), 0, 0., -2.]
        prior_high  = [13.0, 1., 1., 1., 1., 13.27, np.log10(1.5e-2), np.log10(1.5e-2), 3., 3., 1.]
    else: 
        prior_low   = [7, 0., 0., 0., 0., 1e-2, np.log10(4.5e-5), np.log10(4.5e-5), 0, 0., -2., 0]
        prior_high  = [13.0, 1., 1., 1., 1., 13.27, np.log10(1.5e-2), np.log10(1.5e-2), 3., 3., 1., 1.]
elif model == 'modelb': 
    if not freez: 
        prior_low   = [6, 0., 0., 0., 0., 1e-2, np.log10(4.5e-5), np.log10(4.5e-5), 0, 0., -2., 0.1, 0.0, 0.1]
        prior_high  = [13.0, 1., 1., 1., 1., 13.27, np.log10(1.5e-2), np.log10(1.5e-2), 3., 3., 1., 15., 0.15, 0.7]
    else: 
        prior_low   = [6, 0., 0., 0., 0., 1e-2, np.log10(4.5e-5), np.log10(4.5e-5), 0, 0., -2., 0, 0.1, 0.0, 0.1]
        prior_high  = [13.0, 1., 1., 1., 1., 13.27, np.log10(1.5e-2), np.log10(1.5e-2), 3., 3., 1., 1., 15.0, 0.15, 0.7]

 
assert len(prior_low) == theta_train.shape[1], "prior and data dimension mismatch"  

flo = F.Flow(device=device) 
flo.set_prior('uniform', low=prior_low, high=prior_high)
##################################################################################
# OPTUNA
##################################################################################
output_dir = os.path.join(U.data_dir(), 'qphi', model) 

n_trials    = 1000
if not freez: 
    study_name  = 'qphi.%s%s.%s.theta.nmgysigz' % (model, study, bands)
else: 
    study_name  = 'qphi.%s%s.%s.thetaz.nmgysig' % (model, study, bands)

n_jobs     = 1
if not os.path.isdir(os.path.join(output_dir, study_name)): 
    os.system('mkdir %s' % os.path.join(output_dir, study_name))
storage    = 'sqlite:///%s/%s/%s.db' % (output_dir, study_name, study_name)
n_startup_trials = 20

n_blocks_min, n_blocks_max = 4, 20
n_transf_min, n_transf_max = 4, 20
n_hidden_min, n_hidden_max = 64, 1024 
p_drop_min, p_drop_max = 0., 1.


def Objective(trial):
    ''' bojective function for optuna 
    '''
    # Generate the model                                         
    n_blocks = trial.suggest_int("n_blocks", n_blocks_min, n_blocks_max)
    n_transf = trial.suggest_int("n_transf", n_transf_min,  n_transf_max)
    n_hidden = trial.suggest_int("n_hidden", n_hidden_min, n_hidden_max, log=True)
    p_drop = trial.suggest_float("p_drop", p_drop_min, p_drop_max)
    

    nde = flo.nde(
            architecture='maf', 
            n_hidden=n_hidden, 
            n_transf=n_transf, 
            n_blocks=n_blocks, 
            p_drop=p_drop, 
            use_batch_norm=True)

    flo.train_flow(theta_train, X_train, 
            nde=nde, # nde we're training
            summary_writer=SummaryWriter('%s/%s/%s.%i' % (output_dir, study_name, study_name, trial.number)), 
            training_batch_size=500)
    best_valid_log_prob = flo.anpe._summary['best_validation_log_prob'][0]
    flo.anpe._summary_writer.add_hparams(
            {'n_blocks': n_blocks, 'n_transf': n_transf, 'n_hidden': n_hidden, 'p_drop': p_drop},
            {'best_valid_log_prob': best_valid_log_prob}
            )

    # save trained flow
    fqphi   = os.path.join(output_dir, study_name, '%s.%i.pt' % (study_name, trial.number))
    torch.save(flo.flow, fqphi)
        
    return -1*best_valid_log_prob


sampler     = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)
study       = optuna.create_study(study_name=study_name, sampler=sampler, storage=storage, directions=["minimize"], load_if_exists=True) # , "minimize", "minimize"

study.optimize(Objective, n_trials=n_trials, n_jobs=n_jobs)
print("  Number of finished trials: %i" % len(study.trials))
