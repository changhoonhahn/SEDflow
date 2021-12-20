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
        "#SBATCH --time=47:59:59", 
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


def valid(sample, itrain, nhidden, nblocks):
    ''' validate trained ANPE 
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J v.anpe.%s.%ix%i.%i" % (sample, nhidden, nblocks, itrain), 
        "#SBATCH --partition=general",
        "#SBATCH --time=11:59:59", 
        "#SBATCH --export=ALL", 
        "#SBATCH --output=o/valid.anpe.%s.%ix%i.%i.o" % (sample, nhidden, nblocks, itrain), 
        "#SBATCH --mail-type=all", 
        "#SBATCH --mail-user=chhahn@princeton.edu", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate torch-env", 
        "",
        "python /home/chhahn/projects/SEDflow/bin/train/valid.py %s %i %i %i" % (sample, itrain, nhidden, nblocks),
        "",
        'now=$(date +"%T")', 
        'echo "end time ... $now"', 
        ""]) 

    # create the slurm script execute it and remove it
    f = open('_valid.slurm','w')
    f.write(cntnt)
    f.close()
    os.system('sbatch _valid.slurm')
    os.system('rm _valid.slurm')
    return None 



#for i in range(5): 
#    anpe('toy', i, 500, 5)
#    time.sleep(10) 

#for i in range(5): 
#    anpe('toy', i, 100, 10)
#    time.sleep(10) 

#for i in [1]: 
#    anpe('toy', i, 500, 10)

for i in range(5):
    anpe('toy', i, 500, 15)

#for i in range(5): 
#    valid('toy', i, 500, 5)
#    valid('toy', i, 100, 10)

#for i in [1]: 
#    valid('toy', i, 500, 10)
