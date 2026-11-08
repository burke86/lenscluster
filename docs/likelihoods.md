# Likelihoods

The data are the observed image positions. Every observed image $\theta_i$ is ray-traced to its source plane,
$b_i = \theta_i - \alpha(\theta_i)$, and the likelihood compares the images of each family with one source position
$\beta$. The likelihoods differ in how they model the residuals, that is, which errors they assume besides the
catalog astrometry.

Choose a likelihood with `LikelihoodConfig.likelihood_mode` and `gp_kernel`:

| name | `likelihood_mode` | `gp_kernel` | model |
|---|---|---|---|
| GRF (production) | `gp_image_plane` | `knee` | Gaussian random field of unmodeled mass + image-plane GP |
| image-plane GP | `gp_image_plane` | `image_plane` | image-plane GP within each family |
| smoothed (library default) | `smoothed` | | per-image source-plane floors |

In all three, the source positions are sampled as whitened offsets from the conditional source centroid, and the
conditional expectation of the absolute Jacobian determinant normalizes the density.

## GRF of unmodeled mass (`grf_gp`)

The likelihood of the production fits. The residual of image $i$ (family redshift weight $w_i$, magnification matrix $A_i$) is

$$
F_i = b_i - \beta_{k(i)} = w_i\, \delta\alpha(\theta_i) + A_i\, \big(d(\theta_i) + e_i\big),
$$

with three terms:

$\delta\alpha$
: The deflection error of mass the parametric model lacks: a curl-free Gaussian random field on the lens potential
  with the convergence power spectrum $P_\kappa(k) = P_0 / [1 + (k/k_t)^\gamma]$ between fixed wavelength cutoffs. Its
  amplitude $\sigma_\alpha$ is the rms deflection error per component for a source at the reference redshift
  (`gp_reference_redshift`, 2). The knee wavelength $\lambda_t = 2\pi/k_t$ sets the scale of the unmodeled mass.

$d$
: A correlated image-plane displacement with an isotropic Gaussian kernel of length $\ell$ and amplitude
  $\sigma_{\mathrm{shared}}$ (for example, measurement systematics shared by nearby images).

$e_i$
: The independent position error, $\sigma_{\mathrm{inst},i}^2 + \sigma_{\mathrm{white}}^2$.

The covariance of the stacked residuals of a block of images is

$$
C = W K_\alpha W + B\,[\sigma_{\mathrm{sh}}^2 K_{\mathrm{img}} + \mathrm{diag}(\sigma_{\mathrm{inst},i}^2 + \sigma_w^2)]\,B^T + \sigma_{\mathrm{fl}}^2 I,
$$

where $K_\alpha$ holds the longitudinal and transverse correlations of the deflection field (Hankel transforms of
$P_\kappa$, tabulated) and $\sigma_{\mathrm{fl}}$ (`gp_floor_arcsec`, 1.5 mas) is a small smoothness floor.

### Covariance blocks

`gp_blocks` sets which images share one field:

`all`
: One shared mass field over every image (default for full runs). The field-wide translation, convergence and shear
  are projected out (`gp_knee_project_modes`), because the sources and the external shear absorb them.

`system`
: An independent field per system (all knots of one galaxy), with no projection. About five times cheaper per
  gradient, and useful for quick looks. Its posterior differs from the shared field's, so use it for quick tests.

`family`
: An independent field per family.

The init search runs on system blocks either way; the chain-start step and NUTS use the configured blocks.

### Hyperparameters

The sampled hyperparameters are $\sigma_\alpha$, the knee $\lambda_t$, the image-plane $\ell$,
$\sigma_{\mathrm{shared}}$ and $\sigma_{\mathrm{white}}$; the slope $\gamma$ is fixed at 4 unless
`gp_knee_gamma_prior` and `gp_knee_gamma_sigma` free it. The production fits use these settings:

```python
from lenscluster.config import LikelihoodConfig, LogNormalPriorConfig

grf_likelihood = LikelihoodConfig(
    likelihood_mode="gp_image_plane",
    gp_kernel="knee",
    gp_blocks="all",                                            # "system" for quick looks
    gp_knee_project_modes=("translation", "convergence", "shear"),  # only with gp_blocks="all"
    sigma_sel_prior=None,
    gp_knee_gamma=4.0,
    gp_knee_gamma_prior=None,                                   # (2.0, 6.0) with gp_knee_gamma_sigma=0.5 samples it
    gp_knee_scales_arcsec=(0.2, 100.0),
    gp_knee_prior=LogNormalPriorConfig(median=6.0, log_sigma=0.6, lower=2.5, upper=8.0),
    gp_knee_start_arcsec=6.0,
    gp_sigma_alpha_prior=LogNormalPriorConfig(median=0.08, log_sigma=0.5, upper=0.5),
    gp_sigma_alpha_start_arcsec=0.08,
    gp_knee_ell_prior=LogNormalPriorConfig(median=1.0, log_sigma=0.4, lower=0.4, upper=3.0),
    gp_knee_ell_start_arcsec=1.0,
)
config = replace(config, likelihood=grf_likelihood)
```

`gp_knee_fixed_arcsec` holds the knee at one value instead of sampling it.

The knee prior keeps the field at the galaxy scale: left free, the knee moved to 14–25" and absorbed the halo-scale
misfit. A slope of 2 corresponds to truncated isothermal lumps, 4 to NFW-like cusps, and 6 to cored lumps.

## Image-plane GP (`gp_image_plane`)

The image-plane GP alone, per family, with no source-plane floor. Within a family the image-plane displacements are
correlated:

$$
S = \sigma_{\mathrm{shared}}^2 (K \otimes I) + (\sigma_{\mathrm{inst}}^2 + \sigma_{\mathrm{white}}^2) I, \qquad
K_{ij} = \exp\!\left(-\frac{|\theta_i - \theta_j|^2}{2\ell^2}\right),
$$

and the stacked source-plane residuals have covariance $C = B S B^T + \sigma_{\mathrm{fl}}^2 I$ with
$B = \mathrm{blockdiag}(A_i)$. One Cholesky factor of $C$ per family gives the Gaussian term and the conditional
source position.

The sampled scales are $\sigma_{\mathrm{shared}}$ (log-normal, median 0.15", truncated at 1"), $\sigma_{\mathrm{white}}$
and $\ell$ (`gp_correlation_length_prior`, log-normal, median 7", truncated at 30"). Set
`gp_correlation_length_prior=None` to fix $\ell$ at `gp_correlation_length_arcsec`.

## Smoothed local likelihood (`smoothed`)

The library default. Each image's source-plane residual is compared with a Gaussian of covariance

$$
C_i = \sigma_{\mathrm{meas}}^2 A_i A_i^T + \sigma_{\mathrm{def}}^2 I + \sigma_{\mathrm{sel}}^2 R_i, \qquad
R_i = (I + A_i A_i^T / \lambda_{\mathrm{sel}}^2)^{-1},
$$

the propagated measurement error with $\sigma_{\mathrm{meas}}^2 = \sigma_{\mathrm{inst}}^2 + \sigma_{\mathrm{extra}}^2$,
an isotropic residual-deflection scatter $\sigma_{\mathrm{def}}$, and a feature-selection term with stretching weight
$R_i$. $\sigma_{\mathrm{extra}}$, $\sigma_{\mathrm{def}}$, $\sigma_{\mathrm{sel}}$ and $\lambda_{\mathrm{sel}}$ are
sampled. Set `sigma_sel_prior=None` to drop the selection term (as the production fits do) or `sigma_extra_prior=None` to drop
$\sigma_{\mathrm{extra}}$.

## Astrometric errors

$\sigma_{\mathrm{inst}}$ comes from each image's `sigma` in the catalog, otherwise from
`ImageConstraintsConfig.sigma_inst_arcsec`. `LikelihoodConfig.sigma_inst_arcsec` overrides it for every image.
The production fits set `ImageConstraintsConfig(sigma_inst_arcsec=0.001)`, a 1 mas floor, so that the white scatter $\sigma_{\mathrm{white}}$ is inferred rather than assumed; the
"chi-square (catalog errors)" line of the run summary is not meaningful at that value.

## Other terms

Member kinematics
: With `MemberPopulationConfig.kinematics_sources`, the measured dispersions add one log-normal term per member (see
  {doc}`configuration`).

Far-field barrier
: A smooth log-barrier keeps far-field members outside `far_field_cut_radii` of every image during sampling.

`source-anisotropic`, `source-isotropic`
: Source-plane likelihoods kept for diagnostics only.
