NASA-Sloan Atlas (NSA) ``SEDflow`` Value-Added Catalog
======================================================

This probabilistic  value-added catalog provides detailed physical properties
for 33,884 galaxies in the NASA-Sloan Atlas (http://www.nsatlas.org/). 
The properties are inferred from optical photometry in the *u, g, r, i, z*
bands using ``SEDflow``, an accelerated Bayesian SED modeling method that 
combines the |provabgs|_ ``PROVABGS`` SED model with Amortized Neural Posterior 
Estimation. For more details on this catalog and ``SEDflow`` see |sedflow|_.

For each galaxy, the catalog provides 10,000 samples drawn from the posteriors of
::

    logMstar : 
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

    dust1 : 
        birth cloud optical depth

    dust2 : 
        diffuse dust optical depth

    dust_index : 
        Calzetti (2001) dust index

For more details on the galaxy properties, see |provabgs|_. 


Download
--------
Download the catalog: 10.5281/zenodo.6337945

The catalog is in hdf5 format and can be read using https://www.h5py.org/.
::

    import h5py 
    
    f = h5py.File('nsa.sedflow.hdf5', 'r') 
    
    # print data columns 
    print(f.keys())
    
    # read stellar mass 
    logm = f['logMstar'][...]

    f.close()


.. _provabgs: https://ui.adsabs.harvard.edu/abs/2020ApJS..250....2V/abstract/
.. |provabgs| replace:: Hahn *et al.* (2022a) 

.. _sedflow: https://ui.adsabs.harvard.edu/abs/2020ApJS..250....2V/abstract/
.. |sedflow| replace:: Hahn & Melchior (2022) 

