'''

python script to deploy slurm jobs for deploying ANPE

'''
import os, sys
import numpy as np 


def deploy_provabgs(i0, i1, sample='toy', itrain=2, nhidden=500, nblocks=15, niter=5000):
    ''' deploy provabgs on NSA galaxies for which SEDflow failed on 
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J nsa_fail.mcmc.%i_%i" % (i0, i1), 
        "#SBATCH --partition=general",
        "#SBATCH --time=23:59:59", 
        "#SBATCH --export=ALL", 
        "#SBATCH --output=o/nsa_fail.mcmc.%i_%i.o" % (i0, i1), 
        "#SBATCH --mail-type=all", 
        "#SBATCH --mail-user=chhahn@princeton.edu", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate torch-env", 
        "",
        "srun python /home/chhahn/projects/SEDflow/bin/nsa_fail/nsa_fail.py %s %i %i %i %i %i %i" % (sample, itrain, nhidden, nblocks, niter, i0, i1),
        "",
        'now=$(date +"%T")', 
        'echo "end time ... $now"', 
        ""]) 

    # create the slurm script execute it and remove it
    f = open('_nsa.slurm','w')
    f.write(cntnt)
    f.close()
    os.system('sbatch _nsa.slurm')
    os.system('rm _nsa.slurm')
    return None 

igals = np.load('/scratch/network/chhahn/sedflow/nsa_fail/fail.igals.npy')
ngals = len(igals) 
nbatch = int(np.ceil(ngals / 10))

#for ibatch in range(nbatch): 
#    deploy_provabgs(ibatch*10, np.min([(ibatch+1)*10, ngals-1]))
deploy_provabgs(130, 140)
deploy_provabgs(110, 120)
deploy_provabgs(270, 280)
