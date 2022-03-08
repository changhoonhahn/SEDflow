# `SEDflow`: Accelerated Bayesian SED modeling
`SEDflow` is method for  Bayesian SED modeling that uses Amortized Neural Posterior Estimation (ANPE), a simulation-based inference method that employs neural networks to estimate the posterior probability distribution over the full range of observations. 
Once trained, SEDflow requires no additional model evaluations to estimate the posterior.
`SEDflow` takes _∼1 second per galaxy_ to obtain the posteriors of the [Hahn et al. (2022a)](https://ui.adsabs.harvard.edu/abs/2022arXiv220201809H/abstract) SED model parameters, all of which are in excellent agreement with traditional Markov Chain Monte Carlo sampling results.

For additional details on `SEDflow` see [documentation](https://changhoonhahn.github.io/SEDflow/current/) and [Hahn & Melchior (2022)]().
