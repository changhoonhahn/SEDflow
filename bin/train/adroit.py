'''

python script to deploy slurm jobs for constructing training set for speculator

'''
import os, sys
import time 


def anpe(sample, itrain, nhidden, nblocks, gpu=False):
    ''' deploy ANPE training 
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J anpe.%s.%ix%i.%i" % (sample, nhidden, nblocks, itrain), 
        "#SBATCH --partition=general",
        "#SBATCH --time=23:59:59", 
        "#SBATCH --export=ALL", 
        "#SBATCH --output=o/anpe.%s.%ix%i.%i.o" % (sample, nhidden, nblocks, itrain), 
        "#SBATCH --mail-type=all", 
        "#SBATCH --mail-user=chhahn@princeton.edu", 
        ["", "#SBATCH --gres=gpu:1"][gpu], 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate torch-env", 
        "",
        "python /home/chhahn/projects/SEDflow/bin/train/anpe.py %s %i %i %i" % (sample, itrain, nhidden, nblocks),
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


for i in range(10): 
    anpe('toy', i, 100, 5)
    time.sleep(10) 

