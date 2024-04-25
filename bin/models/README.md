# training SEDs 
scripts for constructing the training SEDs with different setups 

## Model A 
`modela.py`: standard `provabgs` setup as specified in the mock challenge
paper. The SEDs will span the observed-frame wavelength range of 0.1 to 30 micron 
and redshift 0 < z < 1. This model, however, doesn't include dust emission properly. 

## Model B 
`modelb.py`: standard `provabgs` setup plus nebular emission and dust emission. Has 3 
additional parameters. The SEDs will span the observed-frame wavelength range of 0.1 to 30 micron 
and redshift 0 < z < 1. This model, however, doesn't include dust emission properly. 

### photometric uncertainties  
$ugrizJ$ photometry: we'll use the magnitude uncertainties from Figure 2 of
[Graham et al. (2020)](https://iopscience.iop.org/article/10.3847/1538-3881/ab8a43/pdf). 

$grzW1W2$ photometry: noise is applied in nanomaggies using a uniformly sampled
standard deviation over the range set by BGS: 
- $0.010 < \sigma_g < 5$ 
- $0.019 < \sigma_r < 20$
- $0.047 < \sigma_z < 15$
- $0.11 < \sigma_{W1} < 10$
- $0.23 < \sigma_{W2} < 15$


# work flow
Below is the workflow for training SEDflow: 

1. Generate SEDs (perlmutter). `1_modela.py` and `1_modelb.py`. For demo see `1_nb_modela.ipynb`
2. Generate photometry for specified photometric bandpasses (perlmutter): `2_fm_photo.py`. For demo see `2_nb_fm_photo.ipynb`. 
3. Forward model noise specific to some observation: `3_fm_noise.ipynb`. In `3_nb_grzW1W2_noise.ipynb` we examine the noises in DESI to set the noise model.
4. Compare forward modled synthetic photometry to observations to ensure sufficient coverage (perlmutter): `4_check_data.ipynb`.
5. Train ANPEs (della): `5_anpe.py` or `5_anpe_della.py`. For demo see `5_nb_anpe_adroit.ipynb`
6. Validate ANPEs (della): `6_qphi_valid_della.ipynb`. Check optuna hyperparameter optimization results: `6_nb_optuna_della.ipynb`. 
