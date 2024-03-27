'''

utility functions

'''
import os
import glob
import numpy as np 

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def data_dir(): 
    ''' get main data directory where the files are stored for whichever machine I'm on 
    '''
    dat_dirs = [
            '/pscratch/sd/c/chahah/sedflow/',  # perlmutter
            '/tigress/chhahn/sedflow/', # tiger
            '/scratch/network/chhahn/sedflow/', # adroit 
            '/Users/chahah/data/sedflow/' # mbp
            ]
    for _dir in dat_dirs: 
        if os.path.isdir(_dir): return _dir


def read_best_qphi(study_name, n_ensemble=5, device='cpu', name='modela'): 
    dat_dir = os.path.join(data_dir(), 'qphi', name) 

    fevents = glob.glob(os.path.join(dat_dir, '%s/*/events*' % study_name))

    events, best_valid = [], []
    for fevent in fevents: 
        ea = EventAccumulator(fevent)
        ea.Reload()

        try: 
            best_valid.append(ea.Scalars('best_validation_log_prob')[0].value)
            events.append(fevent)
        except: 
            pass #print(fevent)

    best_valid = np.array(best_valid)
    print('%i models trained' % np.max([int(os.path.dirname(event).split('.')[-1]) 
        for event in events]))
    
    i_models = [int(os.path.dirname(events[i]).split('.')[-1]) for i 
            in np.argsort(best_valid)[-n_ensemble:][::-1]]
    print(i_models) 
    
    qphis = []
    for i_model in i_models: 
        fqphi = os.path.join(dat_dir, '%s/%s.%i.pt' % (study_name, study_name, i_model))
        qphi = torch.load(fqphi, map_location=device)
        qphis.append(qphi)

    return qphis
