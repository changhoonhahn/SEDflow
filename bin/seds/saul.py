'''

script for deploying job on perlmutter


'''
import os, sys


def modela_sed(nsample, seed):
    ''' create slurm script and then submit
    '''
    cntnt = '\n'.join([
        "#!/bin/bash",
        "#SBATCH --qos=regular",
        "#SBATCH --time=01:59:59",
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


def modela_photo(seed, bands): 
    cntnt = '\n'.join([
        "#!/bin/bash",
        "#SBATCH --qos=debug",
        "#SBATCH --time=00:09:59",
        "#SBATCH --constraint=cpu",
        "#SBATCH -N 1",
        "#SBATCH -J modela%i_%s" % (seed, bands),
        "#SBATCH -o ofiles/modela%i_%s.o" % (seed, bands),
        "",
        'now=$(date +"%T")',
        'echo "start time ... $now"',
        "",
        "source ~/.bashrc",
        "conda activate gqp",
        "",
        "python /global/homes/c/chahah/projects/SEDflow/bin/seds/fm_photo.py modela %i %s" % (seed, bands),
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



# run 2024,03,22
#for i in range(10): 
#    modela_sed(100000, i)

for i in [1]: #range(1, 10): 
    #modela_photo(i, 'grzW1W2') 
    modela_photo(i, 'ugrizJ') 