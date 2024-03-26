# training SEDs 
scripts for constructing the training SEDs with different setups 

## Model A 
`modela.py`: standard `provabgs` setup as specified in the mock challenge
paper. The SEDs will span the observed-frame wavelength range of 0.1 to 30 micron 
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
