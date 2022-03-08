.. sedflow documentation master file, created by
   sphinx-quickstart on Mon Mar  7 10:43:42 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _datamodel:

Accelerated Bayesian SED Modeling
=================================

State-of-the-art SED analyses use a Bayesian framework to infer the physical properties of 
galaxies from observed photometry or spectra. 
They require sampling from a high-dimensional space of SED model parameters and take 
>10-100 CPU hours per galaxy.  
This makes them them practically infeasible for analyzing the billions of galaxies that will 
be observed by upcoming galaxy surveys (e.g. DESI, PFS, Rubin, Webb, and Roman).

``SEDflow`` is method for scalable Bayesian SED modeling that uses Amortized Neural 
Posterior Estimation (ANPE), a simulation-based inference method that employs neural networks 
to estimate the posterior probability distribution over the full range of observations. 
Once trained, ``SEDflow`` requires no additional model evaluations to estimate the posterior. 
``SEDflow`` takes ∼1 second per galaxy to obtain the posteriors of the 12 |provabgs|_ SED 
model parameters, all of which are in excellent agreement with traditional Markov Chain 
Monte Carlo sampling results.

``PROVABGS`` SED Model
------------------
``SEDflow`` applies ANPE to Bayesian galaxy SED modeling using the 
recent |provabgs|_ SED model, the state-of-the-art SPS model of the |desi|_ 
PRObabilistic Value-Added Bright Galaxy Survey (``PROVABGS``). 
The SED of a galaxy is modeled as a composite of stellar populations defined by 
stellar evolution theory, its star formation and chemical enrichment histories 
(SFH and ZH), and dust attenuation. 
The |provabgs|_ model, in particular, utilizes a non-parametric SFH with a 
starburst, a non-parametric ZH that varies with time, and a flexible dust 
attenuation prescription.

NSA `SEDflow` Catalog
---------------------
We apply ``SEDflow`` to 33,884 galaxies in the NASA-Sloan Atlas and construct 
a probabilistic value-added catalog. 
For more details on the catalog and how to download it, see [:ref:`datamodel`] 

Attribution
-----------
Please cite |sedflow|_ if you use the SEDflow NSA catalog in your research.

.. _provabgs: https://ui.adsabs.harvard.edu/abs/2020ApJS..250....2V/abstract/
.. |provabgs| replace:: Hahn *et al.* (2022a) 

.. _sedflow: https://ui.adsabs.harvard.edu/abs/2020ApJS..250....2V/abstract/
.. |sedflow| replace:: Hahn & Melchior (2022) 

.. _desi: http://desi.lbl.gov/
.. |desi| replace:: DESI 


.. toctree::
   :maxdepth: 1
    
   index
   datamodel 
