'''

utility functions

'''
import os


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
