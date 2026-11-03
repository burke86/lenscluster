"""Plot saved GRF convergence power with Fourier normalization; no sampling."""
from pathlib import Path
import sys,json,argparse
import numpy as np
from scipy.integrate import simpson
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from lenscluster.plot_style import PAPER_STYLE

def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();r=a.run.resolve()
 cfg=json.loads((r/'final/artifacts/config.json').read_text())['likelihood']
 specs=json.loads((r/'init/artifacts/state.json').read_text())['parameter_specs'];names=[s['name'] for s in specs]
 d=np.load(r/'final/artifacts/posterior.npz',allow_pickle=True);phys=d['physical'];best=int(np.nanargmax(d['log_prob']))
 knee=phys[:,names.index('images.gp_knee')];sigma=phys[:,names.index('images.gp_sigma_alpha')]
 assert cfg['gp_knee_spectrum']=='knee' and cfg['gp_knee_gamma_prior'] is None
 gamma=cfg['gp_knee_gamma'];small,large=cfg['gp_knee_scales_arcsec'];k=np.geomspace(2*np.pi/large,2*np.pi/small,2401)
 shape=1/(1+(k[None,:]*knee[:,None]/(2*np.pi))**gamma)
 # <alpha_x^2> = integral P_kappa(k) dlnk / pi for the stated 2D Fourier convention.
 integrals=simpson(shape,x=np.log(k),axis=1);P=shape*(np.pi*sigma**2/integrals)[:,None]
 np.testing.assert_allclose(simpson(P,x=np.log(k),axis=1)/np.pi,sigma**2,rtol=1e-12)
 # Independent higher-resolution check of normalization.
 fine=np.geomspace(k[0],k[-1],4801);fi=simpson(1/(1+(fine[None,:]*knee[:,None]/(2*np.pi))**gamma),x=np.log(fine),axis=1)
 assert np.max(abs(fi/integrals-1))<1e-8
 lo,med,hi=np.quantile(P,[.16,.5,.84],axis=0);q=np.quantile(knee,[.16,.5,.84])
 plt.style.use(ROOT/'src/lenscluster/style.mplstyle');plt.rcParams.update(PAPER_STYLE)
 fig,ax=plt.subplots(figsize=(7.1,4.2));color='#0072B2'
 ax.fill_between(k,lo,hi,color=color,alpha=.22,label='Saved draws: 16–84%')
 ax.loglog(k,med,c=color,label='Saved draws: median')
 ax.loglog(k,P[best],c='#D55E00',ls='--',label='Best fit')
 ax.axvline(2*np.pi/q[1],c='.4',lw=.8,ls=':')
 ax.text(.04,.06,rf'$\lambda_b={q[1]:.2f}^{{+{q[2]-q[1]:.2f}}}_{{-{q[1]-q[0]:.2f}}}$ arcsec'+'\n'+rf'$\gamma_\kappa={gamma:g}$ (fixed)',transform=ax.transAxes,fontsize=10)
 ax.set(xlabel=r'Angular wavenumber $k$ [arcsec$^{-1}$]',ylabel=r'Convergence power $P_\kappa(k)$ [arcsec$^2$]',xlim=(k[0],k[-1]))
 ax.legend(loc='upper right',fontsize=8.5)
 def inverse(x):return 2*np.pi/np.maximum(np.asarray(x,dtype=float),1e-300)
 sec=ax.secondary_xaxis('top',functions=(inverse,inverse));sec.set_xlabel(r'Angular wavelength $\lambda=2\pi/k$ [arcsec]');sec.set_xticks([.2,1,10,100],['0.2','1','10','100'])
 fig.subplots_adjust(left=.12,right=.98,bottom=.16,top=.83)
 out=ROOT/'plots/paper/grf_power_spectrum';out.mkdir(parents=True,exist_ok=True);fig.savefig(out/'grf_power_near_10arcsec.pdf');plt.close(fig)
 diag=json.loads((r/'final/artifacts/posterior_diagnostics.json').read_text())
 report=dict(run=str(r),draw=best,break_quantiles=q.tolist(),break_best=float(knee[best]),sigma_alpha_best=float(sigma[best]),gamma_fixed=gamma,
 wavelength_band=[small,large],reference_redshift=cfg['gp_reference_redshift'],divergences=int(d['diverging'].sum()),knee_rhat=diag['r_hat'][names.index('images.gp_knee')],max_rhat=diag['max_r_hat'],
 normalization='sigma_alpha^2 = integral P_kappa dlnk / pi; Fourier covariance = integral d^2k/(2pi)^2 P exp(ik.r)',
 interpretation='Parametric convergence fluctuation spectrum inferred from saved draws; not a direct binned measurement. Outside band power is zero. Bands preliminary due to incomplete convergence. System-block covariance does not correlate different systems.',
 normalization_refinement_max_relative_error=float(np.max(abs(fi/integrals-1))))
 (out/'provenance.json').write_text(json.dumps(report,indent=2));np.savez_compressed(out/'spectrum_draws.npz',k=k,power=P,knee=knee,sigma_alpha=sigma)
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
