``SEDflow`` Training
--------------------
All scripts used to train, validate, and deploy ``SEDflow`` are publicly available in the 
github repo: https://github.com/changhoonhahn/SEDflow . It includes: 

*   ``bin/train/anpe.py``: script used to train ``SEDflow``. Requires ``pytorch`` and the |sbi|_ package. 
*   ``bin/nsa.py``: script used to deploy ``SEDflow`` on optical photometry of NSA galaxies 
*   ``bin/training_data.ipynb``: notebook used to compile and construct training data
*   ``docs/nb/``: notebooks used to generate all the results for |sedflow|_. 

All of the data used to train and test ``SEDflow`` is also publicly available 
at: 10.5281/zenodo.6337945 

*   ``data.v1.*``: training data 
*   ``test.v1.*``: test data 

.. _sbi: https://github.com/mackelab/sbi/
.. |sbi| replace:: ``sbi``

.. _provabgs: https://ui.adsabs.harvard.edu/abs/2020ApJS..250....2V/abstract/
.. |provabgs| replace:: Hahn *et al.* (2022a) 

.. _sedflow: https://ui.adsabs.harvard.edu/abs/2020ApJS..250....2V/abstract/
.. |sedflow| replace:: Hahn & Melchior (2022) 

.. _desi: http://desi.lbl.gov/
.. |desi| replace:: DESI 
