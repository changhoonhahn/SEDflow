'''

python script for deploying job on adroit 

'''
import os, sys
import time 


def anpe(seed, bands, gpu=False):
    ''' deploy ANPE training 
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J modela%i_%s_anpe%s" % (seed, bands, ['', '_gpu'][gpu]),
        "#SBATCH --partition=general",
        "#SBATCH --time=95:59:59", 
        "#SBATCH --export=ALL", 
        "#SBATCH -o ofiles/modela%i_%s_anpe%s.o" % (seed, bands, ['', '_gpu'][gpu]),
        "#SBATCH --mail-type=all", 
        "#SBATCH --mail-user=chhahn@princeton.edu", 
        ["", "#SBATCH --gres=gpu:1"][gpu], 
        ['', "#SBATCH --mem-per-cpu=8G"][gpu], 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate torch-env", 
        "",
        "python /home/chhahn/projects/SEDflow/bin/modela/anpe.py  %s False" % bands,
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

for i in range(5): 
    anpe(i, 'ugrizJ', gpu=False)
    anpe(i, 'grzW1W2', gpu=False)
    #anpe(i, 'ugrizJ', gpu=True)
    #anpe(i, 'grzW1W2', gpu=True)
