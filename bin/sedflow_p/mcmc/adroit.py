'''

python script to deploy slurm jobs for constructing training set for speculator

'''
import os, sys
import time 


def mcmc(i0, i1, niter=5000):
    ''' deploy ANPE training 
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J mcmc.%i_%i" % (i0, i1), 
        "#SBATCH --partition=general",
        "#SBATCH --time=11:59:59", 
        "#SBATCH --export=ALL", 
        "#SBATCH --output=o/mcmc.%i_%i.o" % (i0, i1), 
        "#SBATCH --mail-type=all", 
        "#SBATCH --mail-user=chhahn@princeton.edu", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate torch-env", 
        "",
        "python /home/chhahn/projects/SEDflow/bin/sedflow_p/mcmc/mcmc.py %i %i %i" % (niter, i0, i1), 
        "",
        'now=$(date +"%T")', 
        'echo "end time ... $now"', 
        ""]) 

    # create the slurm script execute it and remove it
    f = open('_anpe.slurm','w')
    f.write(cntnt)
    f.close()
    os.system('sbatch _anpe.slurm')
    os.system('rm _anpe.slurm')
    return None 


#mcmc(0, 9) 
for i in range(1, 10): 
    mcmc(10*i, 10*(i+1)-1)

