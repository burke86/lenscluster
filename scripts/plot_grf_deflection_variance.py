"""Saved convergence/deflection spectra and integrated angular deflection variance.

k_theta is in inverse arcseconds; only the wavelength axis is converted to kpc.
P_alpha,trace = 4 P_kappa/k_theta^2; per-coordinate variance density is P_kappa/pi.
"""
from pathlib import Path
import sys,json
import numpy as np
from scipy.integrate import simpson,cumulative_simpson
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from lenscluster.plot_style import PAPER_STYLE
out=ROOT/'plots/paper/grf_power_spectrum';meta=json.loads((out/'provenance.json').read_text())
state=json.loads((Path(meta['run'])/'init/artifacts/state.json').read_text());scale=state['kpc_per_arcsec']
with np.load(out/'spectrum_draws.npz') as d:k=d['k'];P=d['power'];knee=d['knee'];sigma=d['sigma_alpha']
best=meta['draw'];x=(2*np.pi/k*scale)[::-1];lnx=np.log(x)
def products(p):
 p=np.atleast_2d(p)
 density=p[:,::-1]/np.pi
 cumulative=cumulative_simpson(density,x=lnx,axis=1,initial=0)
 return [((k*k/(2*np.pi))*p)[:,::-1],(p/k**2)[:,::-1],density,cumulative]
curves=products(P);np.testing.assert_allclose(curves[3][:,-1],sigma**2,rtol=1e-11)
assert np.min(np.diff(curves[3],axis=1))>=-1e-15
variants={}
for gamma in [2,6]:
 shape=1/(1+(k*knee[best]/(2*np.pi))**gamma)
 spectrum=shape*np.pi*sigma[best]**2/simpson(shape,x=np.log(k))
 variants[gamma]=products(spectrum)
 np.testing.assert_allclose(variants[gamma][3][0,-1],sigma[best]**2,rtol=1e-11)
plt.style.use(ROOT/'src/lenscluster/style.mplstyle');plt.rcParams.update(PAPER_STYLE)
fig,axes=plt.subplots(2,2,figsize=(7.1,6.4));axes=axes.ravel()
labels=[r'$k_\theta^2P_\kappa/(2\pi)$',r'$P_\kappa/k_\theta^2$ [arcsec$^4$]',r'$d\sigma_{\alpha,x}^2/d\ln\lambda$ [arcsec$^2$]',r'$\sigma_{\alpha,x}^2(\lambda^\prime\leq\lambda)$ [arcsec$^2$]']
for i,(ax,data,ylabel) in enumerate(zip(axes,curves,labels)):
 lo,med,hi=np.quantile(data,[.16,.5,.84],axis=0)
 ax.fill_between(x,lo,hi,color='#0072B2',alpha=.22,label=r'$\gamma_\kappa=4$: 16–84%')
 ax.plot(x,med,c='#0072B2',label=r'$\gamma_\kappa=4$: median')
 ax.plot(x,data[best],c='#D55E00',ls='--',label=r'$\gamma_\kappa=4$: best fit')
 for gamma,c,ls in [(2,'#009E73','-.'),(6,'#CC79A7',':')]:
  ax.plot(x,variants[gamma][i][0],c=c,ls=ls,label=rf'$\gamma_\kappa={gamma}$: comparison')
 ax.set_xscale('log')
 if i<3:ax.set_yscale('log')
 else:
  ax.set_ylim(bottom=0);ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True)
 ax.set(xlim=(x[0],x[-1]),xlabel=r'Wavelength $\lambda$ [kpc]',ylabel=ylabel)
 ax.text(.04,.94,f'({chr(97+i)})',transform=ax.transAxes,va='top')
 ax.axvline(knee[best]*scale,c='.5',lw=.7,ls=':')
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,fontsize=8)
fig.subplots_adjust(left=.12,right=.975,bottom=.09,top=.88,wspace=.38,hspace=.36)
fig.savefig(out/'grf_deflection_power_and_variance.pdf');plt.close(fig)
cum=curves[3][best];total=float(cum[-1]);break_kpc=float(knee[best]*scale)
report=dict(run=meta['run'],best_draw=best,break_wavelength_kpc=break_kpc,
 total_per_coordinate_deflection_variance_arcsec2=total,total_per_coordinate_deflection_rms_arcsec=float(np.sqrt(total)),
 fraction_variance_above_break=float(1-np.interp(np.log(break_kpc),lnx,cum)/total),
 fraction_variance_above_100kpc=float(1-np.interp(np.log(100),lnx,cum)/total),
 half_variance_wavelength_kpc=float(np.exp(np.interp(total/2,cum,lnx))),
 formulas={'angular_wavenumber':'k_theta in arcsec^-1; lambda_kpc=2pi/k_theta * kpc_per_arcsec',
 'panel_b':'P_kappa/k_theta^2; total vector deflection power is four times this',
 'panel_c':'P_kappa/pi; variance contribution per positive logarithmic wavelength interval, per deflection coordinate',
 'panel_d':'Integral from shortest included wavelength up to lambda, computed separately for every draw before quantiles'},
 comparison='gamma=2,6 are not refitted; all comparison spectra share best-fit sigma_alpha, break and band',
 uncertainty='Saved-draw bands remain preliminary because of incomplete convergence; amplitude at source reference redshift 2; source floor and image-plane GP excluded')
(out/'deflection_variance_provenance.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
