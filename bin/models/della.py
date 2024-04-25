'''

python script for deploying job on adroit 

'''
import os, sys
import time 


def anpe(seed, bands, model, study, gpu=False):
    ''' deploy ANPE training 
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J %s%s%i_%s_anpe%s" % (model, study, seed, bands, ['', '_gpu'][gpu]),
        "#SBATCH --time=95:59:59", 
        #"#SBATCH --time=00:29:59", 
        "#SBATCH --export=ALL", 
        "#SBATCH -o ofiles/%s%s%i_%s_anpe%s.o" % (model, study, seed, bands, ['', '_gpu'][gpu]),
        "#SBATCH --mail-type=all", 
        "#SBATCH --mail-user=chhahn@princeton.edu", 
        ["", "#SBATCH --gres=gpu:1"][gpu], 
        ['', "#SBATCH --mem-per-cpu=8G"][gpu], 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate sbi", 
        "",
        "python /home/chhahn/projects/SEDflow/bin/models/anpe_della.py %s False %s %s" % (bands, model, study), 
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
    anpe(i, 'grzW1W2', 'modelb', '.cdf', gpu=False)
    anpe(i, 'grzW1W2', 'modelb', '.cdf', gpu=True)

for i in range(10,30): 
    anpe(i, 'grzW1W2', 'modelb', '.cdf', gpu=False)

#for i in range(10): 
#    anpe(i, 'ugrizJ', '.cdf', gpu=False)
#    anpe(i, 'grzW1W2', '.cdf', gpu=False)
#    anpe(i, 'ugrizJ', '.cdf', gpu=True)
#    anpe(i, 'grzW1W2', '.cdf', gpu=True)
