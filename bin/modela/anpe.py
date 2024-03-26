'''


script to train Neural Posterior Estimators with Optuna hyperparameter optimization


'''
import os, sys 
import numpy as np

import torch
from torch import nn 
from torch.utils.tensorboard.writer import SummaryWriter

import optuna 

from sbi import utils as Ut
from sbi import inference as Inference

from sedflow import data as D
from sedflow import util as U

##################################################################################
# input 
##################################################################################
bands = sys.argv[1] # which bands passes
freez = (sys.argv[2] == True) 
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
x_train, y_train = D.load_modela('train', bands=bands, infer_redshift=freez)
print('Ntrain + Nvalid = %i' % (x_train.shape[0]))

##################################################################################
# set prior 
##################################################################################
if not freez: 
    prior_low   = [7, 0., 0., 0., 0., 1e-2, np.log10(4.5e-5), np.log10(4.5e-5), 0, 0., -2.]
    prior_high  = [12.5, 1., 1., 1., 1., 13.27, np.log10(1.5e-2), np.log10(1.5e-2), 3., 3., 1.]
else: 
    prior_low   = [7, 0., 0., 0., 0., 1e-2, np.log10(4.5e-5), np.log10(4.5e-5), 0, 0., -2., 0]
    prior_high  = [12.5, 1., 1., 1., 1., 13.27, np.log10(1.5e-2), np.log10(1.5e-2), 3., 3., 1., 1.]
assert len(prior_low) == x_train.shape[1], "prior and data dimension mismatch"  

lower_bounds = torch.tensor(prior_low).to(device)
upper_bounds = torch.tensor(prior_high).to(device)

prior = Ut.BoxUniform(low=lower_bounds, high=upper_bounds, device=device)

##################################################################################
# OPTUNA
##################################################################################
output_dir = os.path.join(U.data_dir(), 'qphi', 'modela') 

n_trials    = 1000
if not freez: 
    study_name  = 'qphi.modela.%s.theta.nmgysigz' % bands
else: 
    study_name  = 'qphi.modela.%s.thetaz.nmgysig' % bands 

n_jobs     = 1
if not os.path.isdir(os.path.join(output_dir, study_name)): 
    os.system('mkdir %s' % os.path.join(output_dir, study_name))
storage    = 'sqlite:///%s/%s/%s.db' % (output_dir, study_name, study_name)
n_startup_trials = 100

n_blocks_min, n_blocks_max = 2, 10
n_transf_min, n_transf_max = 2, 10
n_hidden_min, n_hidden_max = 64, 512
n_lr_min, n_lr_max = 5e-6, 1e-3 
p_drop_min, p_drop_max = 0., 1.


def Objective(trial):
    ''' bojective function for optuna 
    '''
    # Generate the model                                         
    n_blocks = trial.suggest_int("n_blocks", n_blocks_min, n_blocks_max)
    n_transf = trial.suggest_int("n_transf", n_transf_min,  n_transf_max)
    n_hidden = trial.suggest_int("n_hidden", n_hidden_min, n_hidden_max, log=True)
    lr = trial.suggest_float("lr", n_lr_min, n_lr_max, log=True) 
    p_drop = trial.suggest_float("p_drop", p_drop_min, p_drop_max)
    
    neural_posterior = Ut.posterior_nn('maf', 
            hidden_features=n_hidden, 
            num_transforms=n_transf, 
            num_blocks=n_blocks, 
            dropout_probability=p_drop, 
            use_batch_norm=True
            )

    anpe = Inference.SNPE(prior=prior,
            density_estimator=neural_posterior,
            device=device, 
            summary_writer=SummaryWriter('%s/%s/%s.%i' % 
                (output_dir, study_name, study_name, trial.number)))

    anpe.append_simulations(
        torch.as_tensor(x_train.astype(np.float32)).to(device),
        torch.as_tensor(y_train.astype(np.float32)).to(device))

    p_x_y_est = anpe.train(
            training_batch_size=50,
            learning_rate=lr, #clip_max_norm=clip_max, 
            show_train_summary=True)

    # save trained NPE  
    qphi    = anpe.build_posterior(p_x_y_est)
    fqphi   = os.path.join(output_dir, study_name, '%s.%i.pt' % (study_name, trial.number))
    torch.save(qphi, fqphi)
        
    best_valid_log_prob = anpe._summary['best_validation_log_prob'][0]

    anpe._summary_writer.add_hparams(
            {'n_blocks': n_blocks, 'n_transf': n_transf, 'n_hidden': n_hidden, 'lr': lr, 'p_drop': p_drop},
            {'best_valid_log_prob': best_valid_log_prob}
            )

    return -1*best_valid_log_prob


sampler     = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials) # multivariate=True)
study       = optuna.create_study(study_name=study_name, sampler=sampler, storage=storage, directions=["minimize"], load_if_exists=True) # , "minimize", "minimize"

study.optimize(Objective, n_trials=n_trials, n_jobs=n_jobs)
print("  Number of finished trials: %i" % len(study.trials))
