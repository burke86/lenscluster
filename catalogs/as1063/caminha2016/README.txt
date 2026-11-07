Strong lensing model version 2.0 for the galaxy cluster Abell S1063.
This model is an update of the model presented in Caminha et al. 2016 A&A 587, A80 using the spectroscopic redshifts of Karman et al. 2017 A&A 599, A28.
When using these redshifts, please acknowledge these two papers.


Lensing maps (magnification, convergence, etc), and additional lenstool output files will be placed in this folder soon.

Input files:
A1063_lenstool_Caminha2016.par
    Lenstool configuration file used for the SL modelling of Abell S1063.


members_95_160_nonan_cuts.cat
    Galaxy members used in the modelling. The magnitudes are F160 Kron measurements using Sextractor. Note the commented galaxy (IDs 1809) which is the BCG and its parameters are free to vary in the modelling.


A1063_obs_arcs_v11.dat
    Multiple image positions used as constrains in the modelling. Notice I selected only image families with spectroscopic confirmation.


bayes.dat
    Full mcmc output from lenstool.

mcmc_convergence.png
    Plot showing the all output chains from lenstool together.

confidence_levels.png
    Confidence levels of all free parameters in the strong lens modelling. The contours correspond to 68%, 95% and 99.7%.
