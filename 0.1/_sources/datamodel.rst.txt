.. _datamodel:

NSA ``SEDflow`` Catalog
=======================

This probabilistic  value-added catalog provides detailed physical properties
for 33,884 galaxies in the NASA-Sloan Atlas (http://www.nsatlas.org/). 
The properties are inferred from optical photometry in the *u, g, r, i, z*
bands using ``SEDflow``, an accelerated Bayesian SED modeling method that 
combines the |provabgs|_ ``PROVABGS`` SED model with Amortized Neural Posterior 
Estimation. For more details on this catalog and ``SEDflow`` see |sedflow|_.

For each galaxy, the catalog provides 10,000 samples drawn from the posteriors of
::

    log_mstar : 
        log10 of stellar mass

    log_sfr_1gyr : 
        log10 of average star formation rate over 1Gyr

    log_Z_MW : 
        log10 of mass-weighted metallicity

    beta1, beta2, beta3, beta4 : 
        coefficients of the non-negative matrix factorization (NMF) 
        star formation history basis functions

    fburst : 
        fraction of stellar mass formed by star burst

    tburst : 
        time of star burst event

    log_gamma1, log_gamma2 : 
        log10 of coefficients of the NMF metallicity history basis 
        functions

    tau_bc : 
        birth cloud optical depth

    tau_ism : 
        diffuse dust optical depth

    n_dust : 
        Calzetti (2001) dust index

For more details on the specific galaxy properties, see |provabgs|_. 

A small fraction fo NSA galaxies have photometry or uncertainties outside 
of the ``SEDflow`` training data. 
For these galaxies, ``SEDflow`` does not produce sensible posteriors. 
We estimate their posteriors using the ``PROVABGS`` SED model with MCMC 
sampling in the same way as |provabgs|_.
We mark these galaxies using:  
::

    sedflow : boolean
        True if posterior was estimated using SEDflow. 
        False if posterior was estimated using MCMC

We also include: 
::

    NSAID : 
        unique ID within the NSA catalog 

    mag_u, mag_g, mag_r, mag_i, mag_z : 
        u, g, r, i, z optical magnitudes derived from NSA catalog's 
        NMGY Galactic-extinction corrected AB photometric flux 
    
    sigma_u, sigma_g, sigma_r, sigma_i, sigma_z: 
        uncertainties of u, g, r, i, z optical photometry in 
        magnitude space
        

Download
--------
Download the catalog ``nsa.sedflow.hdf5``: 10.5281/zenodo.6337945

The catalog is in hdf5 format and can be read using https://www.h5py.org/.
::

    import h5py 
    
    f = h5py.File('nsa.sedflow.hdf5', 'r') 
    
    # print data columns 
    print(f.keys())
    
    # read stellar mass 
    logm = f['log_mstar'][...]

    f.close()


Attribution
-----------
Please cite |sedflow|_ if you use the SEDflow NSA catalog in your research.


.. _provabgs: https://ui.adsabs.harvard.edu/abs/2020ApJS..250....2V/abstract/
.. |provabgs| replace:: Hahn *et al.* (2022a) 

.. _sedflow: https://ui.adsabs.harvard.edu/abs/2020ApJS..250....2V/abstract/
.. |sedflow| replace:: Hahn & Melchior (2022) 
