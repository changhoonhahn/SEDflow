'''

script for deploying job on perlmutter


'''
import os, sys


def deploy_modela(nsample, seed):
    ''' create slurm script and then submit
    '''
    cntnt = '\n'.join([
        "#!/bin/bash",
        "#SBATCH --qos=regular",
        "#SBATCH --time=08:00:00",
        "#SBATCH --constraint=cpu",
        "#SBATCH -N 1",
        "#SBATCH -J modela%i" % seed,
        "#SBATCH -o ofiles/modela%i.o" % seed,
        "",
        'now=$(date +"%T")',
        'echo "start time ... $now"',
        "",
        "source ~/.bashrc",
        "conda activate gqp",
        "",
        "python /global/homes/c/chahah/projects/SEDflow/bin/seds/modela.py %i %i" % (nsample, seed),
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


for i in range(10): 
    deploy_modela(10000, i)
