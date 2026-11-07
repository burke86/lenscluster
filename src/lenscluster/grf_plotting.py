"""Smooth conditional-mean GRF evaluation for figures only.

Sampling and image finding retain gp_knee.MeanDeflectionKnee. This module keeps
its saved conditioning coefficients and low-order subtraction, and replaces the
radial interpolation for display with a C2, curl-free Hermite interpolant.
"""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import BPoly, CubicSpline

from . import gp_knee

PLOT_FIELD_VERSION = 'potential-consistent-hermite-v1'


class SmoothPlotDeflection:
    def __init__(self, layout, tables, coefficients, lam, sigma_alpha2, weights_per_family,
                 gamma=None, modes=()):
        cl, ct = map(np.asarray, gp_knee.knee_curves(tables, jnp.asarray(lam), None if gamma is None else jnp.asarray(gamma)))
        self.r = np.asarray(tables['r_grid'])
        # For a potential field c_L = c_T + r*c_T'. Preserve both tabulated
        # kernel values by prescribing c_T' at every node. Quintic Hermite
        # interpolation also matches second derivatives, so the deflection
        # Jacobian, which contains c_T'', is continuous.
        slope = np.zeros_like(ct)
        slope[1:] = (cl[1:]-ct[1:])/self.r[1:]
        second = CubicSpline(self.r, ct, bc_type=((1, 0.), 'not-a-knot'))(self.r, 2)
        self.ct = BPoly.from_derivatives(self.r, np.column_stack([ct,slope,second]))
        self.ct1, self.ct2 = self.ct.derivative(1), self.ct.derivative(2)
        self.xy = np.asarray(layout['xy'])
        self.mask = np.asarray(layout['mask'], bool)
        self.blocks = np.asarray(layout['block_of_family'], int)
        self.weights = np.asarray(weights_per_family)
        self.amplitude = float(sigma_alpha2)
        n_blocks, size = self.mask.shape
        u = (np.asarray(coefficients)*(self.mask*np.asarray(layout['weights']))[...,None]).reshape(n_blocks,2*size)
        self.affine_offset = np.zeros((n_blocks,2))
        self.affine_jacobian = np.zeros((n_blocks,2,2))
        if modes:
            q, r, centre = map(np.asarray,gp_knee.low_order_projector(layout['xy'],layout['mask'],modes))
            u = u-np.einsum('sin,sn->si',q,np.einsum('sin,si->sn',q,u))
            # Exactly the saved model's low-order subtraction, including its
            # original covariance interpolation. Do not refit the coefficients.
            for block in range(n_blocks):
                d=self.xy[block,:,None]-self.xy[block,None,:]
                radius=np.linalg.norm(d,axis=-1)
                n=d/np.maximum(radius[...,None],1e-30)
                old_cl=np.interp(radius,self.r,cl);old_ct=np.interp(radius,self.r,ct)
                k=self.amplitude*(old_ct[...,None,None]*np.eye(2)+(old_cl-old_ct)[...,None,None]*n[..., :,None]*n[...,None,:])
                k*= (self.mask[block,:,None]&self.mask[block,None,:])[...,None,None]
                ku=np.einsum('imab,mb->ia',k,u[block].reshape(size,2)).reshape(-1)
                g=np.linalg.solve(r[block],q[block].T@ku)
                probe=np.array([[0.,0.],[1.,0.],[0.,1.]])
                pattern=np.asarray(gp_knee.low_order_patterns(probe,centre[block],modes))@g
                self.affine_offset[block]=pattern[0]
                self.affine_jacobian[block]=(pattern[1:]-pattern[0]).T
        self.u=u.reshape(n_blocks,size,2)

    def __call__(self,x,y,families):
        points=np.column_stack([np.asarray(x),np.asarray(y)])
        families=np.asarray(families,int)
        delta=np.empty_like(points);jac=np.empty((len(points),2,2))
        for block in np.unique(self.blocks[families]):
            rows=np.flatnonzero(self.blocks[families]==block)
            xy=self.xy[block,self.mask[block]];u=self.u[block,self.mask[block]]
            d=points[rows,None]-xy[None]
            r=np.linalg.norm(d,axis=-1)
            if np.any(r>self.r[-1]):
                raise ValueError('Plotting GRF separation exceeds the saved covariance table')
            ct=self.ct(r);first=self.ct1(r);second=self.ct2(r)
            safe=np.maximum(r,1e-10)
            f=np.where(r>1e-10,first/safe,self.ct2(0.))
            g=np.where(r>1e-10,(second-first/safe)/safe**2,0.)
            du=np.einsum('pmi,mi->pm',d,u)
            value=np.einsum('pm,ma->pa',ct,u)+np.einsum('pm,pma->pa',f*du,d)
            deriv=(np.einsum('pm,ma,pmc->pac',f,u,d)
                   +np.einsum('pm,pma,mc->pac',f,d,u)
                   +np.einsum('pm->p',f*du)[:,None,None]*np.eye(2)
                   +np.einsum('pm,pma,pmc->pac',g*du,d,d))
            value=self.amplitude*value-self.affine_offset[block]-points[rows]@self.affine_jacobian[block].T
            deriv=self.amplitude*deriv-self.affine_jacobian[block]
            weight=self.weights[families[rows]]
            delta[rows]=value*weight[:,None]
            jac[rows]=deriv*weight[:,None,None]
        return delta[:,0],delta[:,1],jac[:,0,0],jac[:,0,1],jac[:,1,0],jac[:,1,1]


def plot_deflection_correction(evaluator,theta, *, max_separation=None):
    """Opt-in display evaluator; never used by the likelihood or sampler."""
    if not (getattr(evaluator,'gp_knee',False) and hasattr(evaluator,'_knee_mean_inputs')):
        return evaluator.gp_deflection_correction(theta)
    if evaluator.gp_knee_spectrum != 'knee' or evaluator.gp_sigma_alpha_index < 0:
        return evaluator.gp_deflection_correction(theta)
    base=evaluator._padded_evaluator() if evaluator.gp_groups is not None else evaluator
    if not hasattr(base,'_plot_mean_inputs_jit'):
        base._plot_mean_inputs_jit=jax.jit(base._knee_mean_inputs)
    coefficients,scales=base._plot_mean_inputs_jit(jnp.asarray(theta,dtype=jnp.float64))
    scales=np.asarray(scales)
    tables=base.gp_tables
    if max_separation is not None and max_separation > float(tables['r_grid'][-1]):
        # Wide overview corners can exceed the image-to-image separation range.
        # Extend display tables only; conditioning still uses the original tables.
        cached=getattr(base,'_plot_extended_tables',None)
        if cached is None or float(cached['r_grid'][-1]) < max_separation:
            gam=np.asarray(tables['gamma_grid'])
            cached=gp_knee.knee_tables(r_max=float(max_separation)+tables['r_step'],
                r_step=tables['r_step'],lam_min=tables['lam_range'][0],lam_max=tables['lam_range'][1],
                n_lam=len(tables['ln_lam']),scale_min=tables['scales'][0],scale_max=tables['scales'][1],
                gamma=float(gam[0]) if len(gam)==1 else None,
                gamma_grid=None if len(gam)==1 else (float(gam[0]),float(gam[-1]),len(gam)))
            # Preserve the saved interpolation nodes exactly over their original range.
            n=len(tables['r_grid'])
            for key in ('c_l','c_t','d2_l','d2_t'):
                cached[key]=cached[key].at[:n].set(tables[key])
            base._plot_extended_tables=cached
        tables=cached
    return SmoothPlotDeflection(base.gp,tables,coefficients,float(scales[1]),float(scales[0])**2,
        base.gp_weights_per_family,gamma=float(scales[2]) if base.gp_knee_gamma_index>=0 else None,
        modes=tuple(getattr(base,'gp_project_modes',())))
