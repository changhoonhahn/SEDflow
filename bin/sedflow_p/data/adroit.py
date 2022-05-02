'''

python script to deploy slurm jobs for constructing training set for speculator

'''
import time 
import os, sys 


def train_encoder_spec(n_latent, i_model): 
    ''' script for training spectrum encoder 
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J enc_spec.%i.%i" % (n_latent, i_model), 
        "#SBATCH --partition=general",
        "#SBATCH --time=05:59:59", 
        "#SBATCH --export=ALL", 
        "#SBATCH --gres=gpu:1", 
        "#SBATCH --output=o/enc_spec.%i.%i.o" % (n_latent, i_model), 
        "#SBATCH --mail-type=all", 
        "#SBATCH --mail-user=chhahn@princeton.edu", 
        "", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate torch-env", 
        "",
        "python /home/chhahn/projects/SEDflow/bin/sedflow_p/data/encoder_spec.py %i %i" % (n_latent, i_model), 
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


def train_encoder_ivar(n_latent, i_model): 
    ''' deploy training spectrum encoder 
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J enc_ivar.%i.%i" % (n_latent, i_model), 
        "#SBATCH --partition=general",
        "#SBATCH --time=05:59:59", 
        "#SBATCH --export=ALL", 
        "#SBATCH --gres=gpu:1", 
        "#SBATCH --output=o/enc_ivar.%i.%i.o" % (n_latent, i_model), 
        "#SBATCH --mail-type=all", 
        "#SBATCH --mail-user=chhahn@princeton.edu", 
        "", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate torch-env", 
        "",
        "python /home/chhahn/projects/SEDflow/bin/sedflow_p/data/encoder_ivar.py %i %i" % (n_latent, i_model), 
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


def encode_ivar_sdss(n_latent, i_model):
    ''' deploy encode SDSS IVAR
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J sdss.enc_ivar.%i.%i" % (n_latent, i_model), 
        "#SBATCH --partition=general",
        "#SBATCH --time=05:59:59", 
        "#SBATCH --export=ALL", 
        "#SBATCH --gres=gpu:1", 
        "#SBATCH --output=o/sdss.enc_ivar.%i.%i.o" % (n_latent, i_model), 
        "#SBATCH --mail-type=all", 
        "#SBATCH --mail-user=chhahn@princeton.edu", 
        "", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate torch-env", 
        "",
        "python /home/chhahn/projects/SEDflow/bin/sedflow_p/data/process_sdss.py encode_ivar %i %i" % (n_latent, i_model), 
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


def encode_spec_sdss(n_latent, i_model):
    '''
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J sdss.enc_spec.%i.%i" % (n_latent, i_model), 
        "#SBATCH --partition=general",
        "#SBATCH --time=05:59:59", 
        "#SBATCH --export=ALL", 
        "#SBATCH --gres=gpu:1", 
        "#SBATCH --output=o/sdss.enc_spec.%i.%i.o" % (n_latent, i_model), 
        "#SBATCH --mail-type=all", 
        "#SBATCH --mail-user=chhahn@princeton.edu", 
        "", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate torch-env", 
        "",
        "python /home/chhahn/projects/SEDflow/bin/sedflow_p/data/process_sdss.py encode_spec %i %i" % (n_latent, i_model), 
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


def train_nde_noise():
    '''
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J train_nde_noise",
        "#SBATCH --partition=general",
        "#SBATCH --time=23:59:59", 
        "#SBATCH --export=ALL", 
        "#SBATCH --gres=gpu:1", 
        "#SBATCH --output=o/train_nde_noise.o", 
        "#SBATCH --mail-type=all", 
        "#SBATCH --mail-user=chhahn@princeton.edu", 
        "", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate torch-env", 
        "",
        "python /home/chhahn/projects/SEDflow/bin/sedflow_p/data/process_sdss.py train_nde_noise",
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


def sample_nde_noise(ibatch, isplit, arch): 
    ''' deploy generate_trainingdata on adroit 
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J sample.nde_noise.%i.%i" % (ibatch, isplit), 
        "#SBATCH --partition=general",
        "#SBATCH --time=23:59:59", 
        "#SBATCH --mem-per-cpu=12G", 
        "#SBATCH --export=ALL", 
        "#SBATCH --output=o/sample.nde_noise.%i.%i.o" % (ibatch, isplit), 
        "#SBATCH --mail-type=all", 
        "#SBATCH --mail-user=chhahn@princeton.edu", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate torch-env", 
        "",
        "python /home/chhahn/projects/SEDflow/bin/sedflow_p/data/process_train.py sample_nde_noise %i %i %s" % (ibatch, isplit, arch),
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


def apply_nde_noise(ibatch, n_latent, i_model): 
    ''' deploy generate_trainingdata on adroit 
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J apply.nde_noise.%i" % (ibatch), 
        "#SBATCH --partition=general",
        "#SBATCH --time=23:59:59", 
        "#SBATCH --mem-per-cpu=64G", 
        "#SBATCH --export=ALL", 
        "#SBATCH --output=o/_apply.nde_noise.%i.o" % (ibatch), 
        "#SBATCH --mail-type=all", 
        "#SBATCH --mail-user=chhahn@princeton.edu", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate torch-env", 
        "",
        "python /home/chhahn/projects/SEDflow/bin/sedflow_p/data/process_train.py apply_nde_noise %i %i %i" % (ibatch, n_latent, i_model),
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


def encode_spec_train(ibatch, n_latent, i_model): 
    ''' encode noisy training spectra
    '''
    cntnt = '\n'.join([
        "#!/bin/bash", 
        "#SBATCH -J enc_spec.train%i" % (ibatch), 
        "#SBATCH --partition=general",
        "#SBATCH --time=06:59:59", 
        "#SBATCH --mem-per-cpu=32G", 
        "#SBATCH --export=ALL", # "#SBATCH --gres=gpu:1", 
        "#SBATCH --output=o/enc_spec.train%i.o" % (ibatch), 
        "#SBATCH --mail-type=all", 
        "#SBATCH --mail-user=chhahn@princeton.edu", 
        "", 
        'now=$(date +"%T")', 
        'echo "start time ... $now"', 
        "", 
        "source ~/.bashrc", 
        "conda activate torch-env", 
        "",
        "python /home/chhahn/projects/SEDflow/bin/sedflow_p/data/process_train.py encode_spec  %i %i %i" % (ibatch, n_latent, i_model),
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
        "#SBATCH --time=01:59:59", 
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
        "python /home/chhahn/projects/SEDflow/bin/sedflow_p/data/compile_training.py",
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
        "python /home/chhahn/projects/SEDflow/bin/sedflow_p/data/compile_test.py",
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


# generate training data 
#for i in range(10): 
#    training_sed(i, nsample=100000, ncpu=16)
#training_sed(101, nsample=100000, ncpu=16)

#------------------------------------------------
# train encoder and encode SDSS IVAR
#for i in range(1,10): 
#    train_encoder_ivar(5, i)
# 8 is the best encoder for ivar 
#encode_ivar_sdss(5, 8)

# train encoder and encode SDSS Spectra 
#for i in range(10): 
#    train_encoder_spec(10, i)
# 3 is the best encoder for spec
#encode_spec_sdss(10, 3)

# train NDE for p(A_ivar, h_ivar | A_spec, z)
#train_nde_noise()

#------------------------------------------------
# sample sdssivar NDE in 10 different chunks 
#for ibatch in range(1, 10): 
#    for i in range(10): 
#        sample_nde_noise(ibatch, i, 'best')
#for i in range(10): 
#    sample_nde_noise(101, i, 'best')

# apply NDE noise 
#for i in range(10): 
#    apply_nde_noise(i, 5, 8)
#apply_nde_noise(101, 5, 8)

# encoded noisy spectra 
#for i in range(10): 
#    encode_spec_train(i, 10, 3)
#encode_spec_train(101, 10, 3)

compile_training()
compile_test()
