'''

python script to deploy slurm jobs for constructing training set for speculator

'''
import time 
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


def sample_sdssivar_nde(ibatch, isplit): 
    ''' deploy generate_trainingdata on adroit 
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J sample.noise_nde.%i.%i" % (ibatch, isplit), 
        "#SBATCH --partition=general",
        "#SBATCH --time=23:59:59", 
        "#SBATCH --mem-per-cpu=12G", 
        "#SBATCH --export=ALL", 
        "#SBATCH --output=o/_sample.noise_nde.%i.%i.o" % (ibatch, isplit), 
        "#SBATCH --mail-type=all", 
        "#SBATCH --mail-user=chhahn@princeton.edu", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate torch-env", 
        "",
        "python /home/chhahn/projects/SEDflow/bin/sedflow_p/training_data/sample_sdss_ivar_nde.py %i %i"  % (ibatch, isplit),
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


def apply_sdssivar_nde(ibatch): 
    ''' deploy generate_trainingdata on adroit 
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J apply.noise_nde.%i" % (ibatch), 
        "#SBATCH --partition=general",
        "#SBATCH --time=23:59:59", 
        "#SBATCH --mem-per-cpu=64G", 
        "#SBATCH --export=ALL", 
        "#SBATCH --output=o/_apply.noise_nde.%i.o" % (ibatch), 
        "#SBATCH --mail-type=all", 
        "#SBATCH --mail-user=chhahn@princeton.edu", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate torch-env", 
        "",
        "python /home/chhahn/projects/SEDflow/bin/sedflow_p/training_data/apply_sdss_ivar_nde.py %i"  % (ibatch),
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


def encode_training_sed(ibatch): 
    ''' deploy encode_sed_sdss_ivar_nde.py on adroit 
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J encode.%i" % (ibatch), 
        "#SBATCH --partition=general",
        "#SBATCH --time=23:59:59", 
        "#SBATCH --mem-per-cpu=32G", 
        "#SBATCH --export=ALL", 
        "#SBATCH --gres=gpu:1", 
        "#SBATCH --output=o/encode.%i.o" % (ibatch), 
        "#SBATCH --mail-type=all", 
        "#SBATCH --mail-user=chhahn@princeton.edu", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate torch-env", 
        "",
        "python /home/chhahn/projects/SEDflow/bin/sedflow_p/training_data/encode_sed_sdss_ivar_nde.py %i"  % (ibatch),
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

def encode_sdss(): 
    ''' deploy encode_sed_sdss_ivar_nde.py on adroit 
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J encode.sdss", 
        "#SBATCH --partition=general",
        "#SBATCH --time=23:59:59", 
        "#SBATCH --mem-per-cpu=32G", 
        "#SBATCH --export=ALL", 
        "#SBATCH --gres=gpu:1", 
        "#SBATCH --output=o/encode.sdss.o", 
        "#SBATCH --mail-type=all", 
        "#SBATCH --mail-user=chhahn@princeton.edu", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate torch-env", 
        "",
        "python /home/chhahn/projects/SEDflow/bin/training_data/encode_sdss.py",
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


def compile_training(): 
    ''' deploy encode_sed_sdss_ivar_nde.py on adroit 
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J compile.train", 
        "#SBATCH --partition=general",
        "#SBATCH --time=23:59:59", 
        "#SBATCH --mem-per-cpu=32G", 
        "#SBATCH --export=ALL", 
        "#SBATCH --output=o/compile.train.o", 
        "#SBATCH --mail-type=all", 
        "#SBATCH --mail-user=chhahn@princeton.edu", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate torch-env", 
        "",
        "python /home/chhahn/projects/SEDflow/bin/training_data/compile_training.py",
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


def compile_test(): 
    ''' deploy encode_sed_sdss_ivar_nde.py on adroit 
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J compile.train", 
        "#SBATCH --partition=general",
        "#SBATCH --time=23:59:59", 
        "#SBATCH --mem-per-cpu=32G", 
        "#SBATCH --export=ALL", 
        "#SBATCH --output=o/compile.test.o", 
        "#SBATCH --mail-type=all", 
        "#SBATCH --mail-user=chhahn@princeton.edu", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate torch-env", 
        "",
        "python /home/chhahn/projects/SEDflow/bin/sedflow_p/training_data/compile_test.py",
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


#for i in range(10): 
#    training_sed(i, nsample=100000, ncpu=16)
#training_sed(101, nsample=100000, ncpu=16)

# sample sdssivar NDE in 10 different chunks 
#for ibatch in range(2, 10): 
#    for i in range(10): 
#        sample_sdssivar_nde(ibatch, i)
#for i in range(10): 
#    sample_sdssivar_nde(101, i)

#for i in range(10): 
#    apply_sdssivar_nde(i)
#apply_sdssivar_nde(101)

#for i in range(10): 
#    encode_training_sed(i)
#encode_training_sed(101)

#encode_sdss()

#compile_training()

compile_test()
