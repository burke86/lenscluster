"""Re-express the saved 2D convergence spectrum as variance per logarithmic scale."""
from pathlib import Path
import sys,json
import numpy as np
from scipy.integrate import simpson
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from lenscluster.plot_style import PAPER_STYLE

out=ROOT/'plots/paper/grf_power_spectrum'
provenance=json.loads((out/'provenance.json').read_text())
run=Path(provenance['run']);state=json.loads((run/'init/artifacts/state.json').read_text())
scale=float(state['kpc_per_arcsec']);z=float(state['z_lens'])
with np.load(out/'spectrum_draws.npz') as d:
 k=d['k'];P=d['power'];knee=d['knee']
k_phys=k/scale;P_phys=P*scale**2
power=k_phys[None,:]**2*P_phys/(2*np.pi)
np.testing.assert_allclose(power,k[None,:]**2*P/(2*np.pi),rtol=1e-14)
wavelength=2*np.pi/k_phys
# Integrating in log wavelength (in ascending order) gives the same convergence variance.
vk=simpson(power,x=np.log(k_phys),axis=1)
vl=simpson(power[:,::-1],x=np.log(wavelength[::-1]),axis=1)
np.testing.assert_allclose(vk,vl,rtol=1e-12)
# With fixed gamma=4, every draw peaks at k_b, hence wavelength lambda_b.
peak=wavelength[np.argmax(power,axis=1)]
assert np.max(np.abs(np.log(peak/(knee*scale)))) < np.log(k[1]/k[0])
lo,med,hi=np.quantile(power,[.16,.5,.84],axis=0);q=np.quantile(knee*scale,[.16,.5,.84]);best=provenance['draw']
plt.style.use(ROOT/'src/lenscluster/style.mplstyle');plt.rcParams.update(PAPER_STYLE)
fig,ax=plt.subplots(figsize=(7.1,4.2));order=np.argsort(wavelength);x=wavelength[order]
ax.fill_between(x,lo[order],hi[order],color='#0072B2',alpha=.22,label=r'$\gamma_\kappa=4$: 16–84%')
ax.loglog(x,med[order],c='#0072B2',label=r'$\gamma_\kappa=4$: median')
ax.loglog(x,power[best,order],c='#D55E00',ls='--',label=r'$\gamma_\kappa=4$: best fit')
comparison_checks={}
for gamma,color,style in [(2,'#009E73','-.'),(6,'#CC79A7',':')]:
 shape=1/(1+(k*knee[best]/(2*np.pi))**gamma)
 spectrum=shape*np.pi*provenance['sigma_alpha_best']**2/simpson(shape,x=np.log(k))
 recovered_variance=simpson(spectrum,x=np.log(k))/np.pi
 np.testing.assert_allclose(recovered_variance,provenance['sigma_alpha_best']**2,rtol=1e-13)
 comparison_checks[str(gamma)]=float(recovered_variance)
 ax.loglog(x,(k**2*spectrum/(2*np.pi))[order],color=color,ls=style,
           label=rf'$\gamma_\kappa={gamma}$: comparison')
ax.axvline(q[1],c='.4',lw=.8,ls=':')
ax.text(.53,.06,rf'$\lambda_b={q[1]:.1f}^{{+{q[2]-q[1]:.1f}}}_{{-{q[1]-q[0]:.1f}}}$ kpc'+'\n'+rf'$\gamma_\kappa=4$ (fixed), $z_l={z:g}$',transform=ax.transAxes,fontsize=10)
ax.set(xlabel=r'Physical wavelength $\lambda=2\pi/k$ [kpc]',ylabel=r'$\Delta_\kappa^2(k)=k^2P_\kappa(k)/(2\pi)$',xlim=(x[0],x[-1]))
fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=3,fontsize=8)
fig.subplots_adjust(left=.12,right=.98,bottom=.17,top=.80)
fig.savefig(out/'grf_dimensionless_power_vs_wavelength_kpc.pdf');plt.close(fig)
info=dict(run=str(run),lens_redshift=z,kpc_per_arcsec=scale,break_kpc_quantiles=q.tolist(),
 dimensionality=2,definition='Delta_kappa^2=k^2 P_kappa/(2pi); integral over dlnk gives variance of convergence at reference source redshift 2',
 physical_power_conversion='k_phys=k_arcsec/scale; P_phys=P_arcsec*scale^2',
 interpretation='proper transverse wavelength at lens redshift, not a halo truncation radius; preliminary bands from incompletely converged saved draws',
 maximum_variance_integral_relative_error=float(np.max(abs(vl/vk-1))),
 comparison_slopes=[2,6],comparison_knee_arcsec=float(knee[best]),
 comparison_sigma_alpha_arcsec=provenance['sigma_alpha_best'],
 comparison_definition='Not refitted; same best-fit break, per-coordinate deflection scatter, and wavelength cutoffs. Each spectrum renormalized to preserve deflection variance.',
 comparison_recovered_deflection_variances=comparison_checks)
(out/'dimensionless_provenance.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2))
