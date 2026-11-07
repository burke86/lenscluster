"""Differentiable flat-wCDM distances by fixed Gauss-Legendre quadrature.

The production distance helper uses interpolation on a finite table. This
isolated adapter avoids introducing that interpolation error into derivatives
of the cosmography forecast. Accuracy is checked at runtime against Astropy.
"""
import numpy as np
import jax.numpy as jnp

NODES,WEIGHTS=np.polynomial.legendre.leggauss(128)

def comoving(z,h0,omega_m,w0):
    z=jnp.asarray(z);q=(jnp.asarray(NODES)+1)*z[...,None]/2
    inverse_e=1/jnp.sqrt(omega_m*(1+q)**3+(1-omega_m)*(1+q)**(3*(1+w0)))
    return 299792.458/h0*z/2*jnp.sum(jnp.asarray(WEIGHTS)*inverse_e,axis=-1)

def flat_wcdm_lensing_efficiency(z_lens,z_source,h0,omega_m,w0):
    lens=comoving(z_lens,h0,omega_m,w0);source=comoving(z_source,h0,omega_m,w0)
    return jnp.where(source>lens,1-lens/source,jnp.nan)

def kpc_per_arcsec(z,h0=70.,omega_m=.3,w0=-1.):
    return comoving(z,h0,omega_m,w0)/(1+z)*1000*np.deg2rad(1/3600)
