'''

script for deploying job on perlmutter


'''
import os, sys


def desi_lowz(istart, iend, debug=False):
    ''' deploy sedflow on DESI LOWZ data 
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J desi_lowz.%i_%i" % (istart, iend),
        ["#SBATCH --time=03:59:59", "#SBATCH --time=00:29:59"][debug],
        "#SBATCH --export=ALL", 
        "#SBATCH -o o/desi_lowz.%i_%i.o" % (istart, iend),
        "#SBATCH --gres=gpu:1", 
        "#SBATCH --mem-per-cpu=16G", 
        "#SBATCH --mail-type=all", 
        "#SBATCH --mail-user=chhahn@princeton.edu", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate sbi", 
        "",
        "python /home/chhahn/projects/SEDflow/bin/deploy/desi_lowz/desi_lowz.py %i %i" % (istart, iend), 
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


def compile(debug=True):
    ''' deploy sedflow on DESI LOWZ data 
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J desi_lowz_compile",
        ["#SBATCH --time=05:59:59", "#SBATCH --time=00:29:59"][debug],
        "#SBATCH --export=ALL", 
        "#SBATCH -o o/desi_lowz_compile.o",
        "#SBATCH --mem-per-cpu=16G", 
        "#SBATCH --mail-type=all", 
        "#SBATCH --mail-user=chhahn@princeton.edu", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate sbi", 
        "",
        "python /home/chhahn/projects/SEDflow/bin/deploy/desi_lowz/compile.py", 
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


# 102325 total 
#for ichunk in range(11): 
#    desi_lowz(ichunk * 10000, (ichunk+1) * 10000, debug=False) 

compile(debug=False)
