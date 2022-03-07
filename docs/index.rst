.. sedflow documentation master file, created by
   sphinx-quickstart on Mon Mar  7 10:43:42 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Accelerated Bayesian SED Modeling
=================================

State-of-the-art SED analyses use a Bayesian framework to infer the physical properties of galaxies from observed photometry or spectra. 
They require sampling from a high-dimensional space of SED model parameters and take >10-100 CPU hours per galaxy. 
This makes them them practically infeasible for analyzing the billions of galaxies that will be observed by upcoming galaxy surveys (e.g. DESI, PFS, Rubin, Webb, and Roman).
`SEDflow` is method for scalable Bayesian SED modeling that uses Amortized Neural Posterior Estimation (ANPE).
ANPE is a simulation-based inference method that employs neural networks to estimate the posterior probability distribution over the full range of observations. 
Once trained, it requires no additional model evaluations to estimate the posterior. 
We present, and publicly release, SEDflow, an ANPE method to produce posteriors of the recent Hahn et al. (2022) SED model from optical photometry. 
`SEDflow` takes ∼1 second per galaxy to obtain the posterior distributions of 12 model parameters, all of which are in excellent agreement with traditional Markov Chain Monte Carlo sampling results.


NSA SEDflow Catalog
-------------------

We apply `SEDflow` to 33,884 galaxies in the NASA-Sloan Atlas and infer posteriors on stellar mass, star formation rate, 
The posteriors are publicly available at 

Attribution
-----------

Please cite `Hahn & Melchior (2022) <>`_ if you use the SEDflow NSA catalog in your research.


.. toctree::
   :maxdepth: 2

   datamodel 
