# `SEDflow`: Accelerated Bayesian SED modeling
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6345467.svg)](https://doi.org/10.5281/zenodo.6345467)

`SEDflow` is an accelerated Bayesian SED modeling method that uses Amortized Neural Posterior Estimation (ANPE), a simulation-based inference method that employs neural networks to estimate the posterior probability distribution over the full range of observations. Once trained, it requires no additional model evaluations to estimate the posterior.  `SEDflow` takes _∼1 second per galaxy_ to derive posteriors of the [Hahn et al. (2022a)](https://ui.adsabs.harvard.edu/abs/2022arXiv220201809H/abstract) SED model parameters that are in excellent agreement with traditional Markov Chain Monte Carlo sampling results. `SEDflow` is ~100,000\times faster than convetional methods. 

For additional details on `SEDflow` see [documentation](https://changhoonhahn.github.io/SEDflow/current/) and [Hahn & Melchior (2022)](https://arxiv.org/abs/2203.07391).
