'''

script for deploying job on perlmutter


'''
import os, sys


def desiy1_bgs(block, debug=False):
    ''' deploy sedflow on DESI Y1 BGS data
    '''
    cntnt = '\n'.join([
        "#!/bin/bash",
        ["#SBATCH --qos=debug", "#SBATCH --qos=regular"][~debug], 
        ["#SBATCH --time=00:29:59", "#SBATCH --time=05:59:59"][~debug],
        "#SBATCH --constraint=cpu",
        "#SBATCH -N 1",
        "#SBATCH -J desiy1.bgs%i" % block,
        "#SBATCH -o o/desiy1.bgs%i.o" % block,
        "",
        'now=$(date +"%T")',
        'echo "start time ... $now"',
        "",
        "source ~/.bashrc",
        "conda activate gqp",
        "",
        "python /global/homes/c/chahah/projects/SEDflow/bin/deploy/desiy1/desiy1_bgs.py %i" % block, 
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

def desiy1_lrg_act(istart, iend, debug=False):
    ''' deploy sedflow on DESI Y1 BGS data
    '''
    cntnt = '\n'.join([
        "#!/bin/bash",
        ["#SBATCH --qos=regular", "#SBATCH --qos=debug"][debug], 
        ["#SBATCH --time=14:59:59", "#SBATCH --time=00:29:59"][debug],
        "#SBATCH --constraint=cpu",
        "#SBATCH -N 1", #"#SBATCH --ntasks=1", #"#SBATCH --mem=16GB",
        "#SBATCH -J desiy1.lrg_act.%i_%i" % (istart, iend),
        "#SBATCH -o o/desiy1.lrg_act.%i_%i.o" % (istart, iend),
        "",
        'now=$(date +"%T")',
        'echo "start time ... $now"',
        "",
        "source ~/.bashrc",
        "conda activate gqp",
        "",
        "python /global/homes/c/chahah/projects/SEDflow/bin/deploy/desiy1/desiy1_lrg_act.py %i %i" % (istart, iend), 
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

# run 2024.04.25
#desiy1_bgs(4, debug=True) # test SEDs
#for i in range(58): 
#    desiy1_bgs(i) # test SEDs

# run 2024.06.17
for i in range(101,166): 
    desiy1_lrg_act(i*5000, (i+1)*5000, debug=False)
