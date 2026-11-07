Strong lensing model version 1.0 (2022) for the galaxy cluster MACS J0416 presented in Bergamini et al. 2022.

The mass model separates the diffuse dark-matter and the hot intracluster gas into distinct mass components.
We refer to Bergamini et al. 2022 for additional details.

The full CLASH-VLT spectroscopic catalogue, combining VIMOS and MUSE, can be found in https://sites.google.com/site/vltclashpublic/.

The modelling is performed with the software lenstool (Kneib et al 1996, Jullo et al 2007, Jullo & Kneib 2009) and using the position of multiple images as constraints.



In this repository, we release the lenstool configuration files, the best fitting model and the full MCMC chain. A detailed explanation of the configuration and output files can be found in the lenstool documentation (https://projets.lam.fr/projects/lenstool/wiki/LenstoolManual).


Input files:
Bergamini22_MACS0416.par
    Lenstool configuration file used for the SL modelling of MACS J0416.


CM_cat_MACSJ0416.cat
    Galaxy members used in the modelling. The magnitudes are F160 Kron measurements using Sextractor.


obs_arcs.dat
    Multiple image positions used as constrains in the modelling. Notice we selected only image families with spectroscopic confirmation.


bayes.dat
    Mcmc output from lenstool (sampling phase).


burnin.dat
    Mcmc output from lenstool (burn-in phase).


chires.dat
    Residuals on image positions of the best-fit model.


bestopt.par best.par
    Lenstool configuration files containing the best-fit model of MACS J0416