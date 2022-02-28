'''

python script to deploy slurm jobs for deploying ANPE

'''
import os, sys
import time 


def deploy_provabgs(sample='toy', itrain=2, nhidden=500, nblocks=15, niter=5000, n_cpu=32):
    ''' deploy provabgs on NSA galaxies for which SEDflow failed on 
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J nsa_fail.mcmc", 
        "#SBATCH --partition=general",
        "#SBATCH --time=23:59:59", 
        "#SBATCH --export=ALL", 
        "#SBATCH --output=o/nsa_fail.mcmc.o", 
        "#SBATCH --mail-type=all", 
        "#SBATCH --mail-user=chhahn@princeton.edu", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate torch-env", 
        "",
        "python /home/chhahn/projects/SEDflow/bin/nsa_fail/nsa_fail.py %s %i %i %i %i %i" % (sample, itrain, nhidden, nblocks, niter, n_cpu),
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

deploy_provabgs()

