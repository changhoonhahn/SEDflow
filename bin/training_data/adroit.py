'''

python script to deploy slurm jobs for constructing training set for speculator

'''
import os, sys 


def training_sed(ibatch, nsample=100000, ncpu=32): 
    ''' deploy generate_trainingdata on adroit 
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J sedflow.sed%i" % ibatch, 
        "#SBATCH --partition=general",
        "#SBATCH --time=00:59:59", 
        '#SBATCH --cpus-per-task=%i' % ncpu, 
        "#SBATCH --export=ALL", 
        "#SBATCH --output=o/sedflow.sed%i.o" % ibatch, 
        "#SBATCH --mail-type=all", 
        "#SBATCH --mail-user=chhahn@princeton.edu", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate torch-env", 
        "",
        "srun python /home/chhahn/projects/SEDflow/bin/training_data/training_sed.py %i %i"  % (nsample, ibatch),
        "",
        'now=$(date +"%T")', 
        'echo "end time ... $now"', 
        ""]) 

    # create the slurm script execute it and remove it
    f = open('_train.slurm','w')
    f.write(cntnt)
    f.close()
    os.system('sbatch _train.slurm')
    os.system('rm _train.slurm')
    return None 


for i in range(1, 10): 
    training_sed(0, nsample=100000, ncpu=16)
training_sed(101, nsample=100000, ncpu=16)
