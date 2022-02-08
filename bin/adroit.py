'''

python script to deploy slurm jobs for deploying ANPE

'''
import os, sys
import time 


def deploy_nsa(ichunk, sample='toy', itrain=2, nhidden=500, nblocks=15):
    ''' deploy trained ANPE on NSA 
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J nsa.anpe.%s.%ix%i.%i.%i" % (sample, nhidden, nblocks, itrain, ichunk), 
        "#SBATCH --partition=general",
        "#SBATCH --time=71:59:59", 
        "#SBATCH --export=ALL", 
        "#SBATCH --output=o/nsa.anpe.%s.%ix%i.%i.%i.o" % (sample, nhidden, nblocks, itrain, ichunk), 
        "#SBATCH --mail-type=all", 
        "#SBATCH --mail-user=chhahn@princeton.edu", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate torch-env", 
        "",
        "python /home/chhahn/projects/SEDflow/bin/nsa.py %s %i %i %i %i" % (sample, itrain, nhidden, nblocks, ichunk),
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

for i in range(34): 
    time.sleep(1)
    deploy_nsa(i)

