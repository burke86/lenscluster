"""Additional synthetic covariance illustrations; no inference or run products."""
from pathlib import Path
import json
import numpy as np
from scipy.special import jv
from numpy.polynomial.legendre import leggauss
from functools import lru_cache
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.backends.backend_pdf import PdfPages
from cmcrameri import cm
from make_covariance_revision import ROOT, PAPER_STYLE, SM, SA, SH, ELL, KNEE, COLORS
OUT=ROOT/'plots/paper/covariance_intuition'

@lru_cache(None)
def nodes():
    x,w=leggauss(1400);lo,hi=np.log(2*np.pi/100),np.log(2*np.pi/.2)
    k=np.exp((hi+lo)/2+(hi-lo)*x/2);w=w/(1+(k/(2*np.pi/KNEE))**4)
    return k,w/w.sum()

def corr(r):
    k,w=nodes();kr=np.asarray(r)[...,None]*k
    return (jv(0,kr)-jv(2,kr))@w,(jv(0,kr)+jv(2,kr))@w

def ellipse(ax,C,color):
    ev,V=np.linalg.eigh(C);ang=np.degrees(np.arctan2(V[1,1],V[0,1]))
    ax.add_patch(Ellipse((0,0),2*np.sqrt(ev[1])*np.sqrt(5.991464547),2*np.sqrt(ev[0])*np.sqrt(5.991464547),angle=ang,fill=False,ec=color,lw=1.3))

def pair():
    fig,axes=plt.subplots(1,3,figsize=(9.3,3.3));r=.6;a=.2
    cl=float(corr(r)[0]);rho=np.exp(-r*r/(2*ELL**2))
    matrices=[SM**2*a*a*np.eye(2),SA**2*np.array([[1,cl],[cl,1]]),SH**2*a*a*np.array([[1,-rho],[-rho,1]])]
    names=['Independent measurement','Shared deflection GRF','Image GP, opposite parity']
    rng=np.random.default_rng(173);checks=[]
    for i,(ax,C,name,color) in enumerate(zip(axes,matrices,names,COLORS)):
        sample=rng.multivariate_normal([0,0],C,1200)
        ax.scatter(*sample.T,s=3,color=color,alpha=.19,rasterized=True,edgecolors='none')
        ellipse(ax,C,color);ax.plot(0,0,'+',c='black',ms=6,mew=.8)
        ax.set(xlim=(-.12,.12),ylim=(-.12,.12),aspect='equal',xlabel=r'Image 1 residual [arcsec]')
        ax.set_ylabel('Image 2 residual [arcsec]' if i==0 else '')
        ax.text(.04,.95,f'({chr(97+i)})',transform=ax.transAxes,va='top')
        ax.plot([],[],color=color,label=name)
        ax.legend(loc='lower left',fontsize=7,handlelength=1.2)
        ax.text(.96,.90,rf'$\rho={C[0,1]/C[0,0]:.2f}$',transform=ax.transAxes,ha='right')
        checks.append({'variance':float(C[0,0]),'rho':float(C[0,1]/C[0,0]),'min_eigenvalue':float(np.linalg.eigvalsh(C).min())})
    fig.subplots_adjust(left=.075,right=.99,bottom=.19,top=.86,wspace=.16)
    return fig,checks

def field():
    axis=np.linspace(-15,15,101);x,y=np.meshgrid(axis,axis);r=np.hypot(x,y);cl,ct=corr(r.ravel());cl=cl.reshape(r.shape);ct=ct.reshape(r.shape)
    nx=np.divide(x,r,out=np.zeros_like(x),where=r>0);ny=np.divide(y,r,out=np.zeros_like(y),where=r>0)
    # Conditional mean given delta alpha(0) = (sigma_alpha,0).
    vx=ct+(cl-ct)*nx*nx;vy=(cl-ct)*nx*ny
    fig,axes=plt.subplots(1,2,figsize=(8.6,3.7))
    for i,(ax,z,label) in enumerate(zip(axes,[vx,vy],[r'$\langle\delta\alpha_x\mid\delta\boldsymbol\alpha(0)=(\sigma_\alpha,0)\rangle/\sigma_\alpha$',r'$\langle\delta\alpha_y\mid\delta\boldsymbol\alpha(0)=(\sigma_\alpha,0)\rangle/\sigma_\alpha$'])):
        im=ax.imshow(z,origin='lower',extent=(-15,15,-15,15),cmap=cm.vik,vmin=-1,vmax=1)
        sl=(slice(5,None,9),slice(5,None,9));ax.quiver(x[sl],y[sl],vx[sl],vy[sl],color='black',angles='xy',scale_units='xy',scale=.42,width=.004)
        ax.plot(0,0,'o',mfc='white',mec='black',ms=5)
        ax.set(xlabel='Offset x [arcsec]',ylabel='Offset y [arcsec]',xlim=(-15,15),ylim=(-15,15))
        ax.text(.04,.95,f'({chr(97+i)})',transform=ax.transAxes,va='top')
        ax.text(.96,.95,'x component' if i==0 else 'y component',transform=ax.transAxes,ha='right',va='top',fontsize=9)
    fig.subplots_adjust(left=.075,right=.87,bottom=.16,top=.89,wspace=.32)
    cb=fig.colorbar(im,cax=fig.add_axes([.90,.22,.018,.61]));cb.set_label('Conditional mean / deflection scatter')
    return fig

def averaging():
    n=np.arange(1,25);spacing=[.2,10.];a=.2
    marginal=SA**2+a*a*(SM**2+SH**2);fig,axes=plt.subplots(1,2,figsize=(8.5,3.4));checks={}
    for d,c,label in zip(spacing,[COLORS[1],COLORS[0]],['Closely spaced: 0.2 arcsec','Widely spaced: 10 arcsec']):
        variance=[]
        for count in n:
            distance=np.abs(np.subtract.outer(np.arange(count),np.arange(count)))*d
            cl,_=corr(distance);C=SA**2*cl+a*a*SH**2*np.exp(-distance**2/(2*ELL**2))+a*a*SM**2*np.eye(count)
            assert np.linalg.eigvalsh(C).min()>0
            variance.append(C.sum()/count**2)
        variance=np.array(variance);checks[str(d)]=variance.tolist()
        axes[0].plot(n,np.sqrt(variance),color=c,label=label)
        axes[1].plot(n,marginal/variance,color=c,label=label)
    axes[0].plot(n,np.sqrt(marginal/n),color='black',ls='--',label='Independent images')
    axes[1].plot(n,n,color='black',ls='--')
    axes[0].set(xlabel='Number of images',ylabel='Uncertainty of mean residual [arcsec]',xlim=(1,24),ylim=(0,.046))
    axes[1].set(xlabel='Number of images',ylabel='Independent-image equivalent',xlim=(1,24),ylim=(0,1.08*max(marginal/np.array(checks['10.0']))))
    for i,ax in enumerate(axes):ax.text(.04,.95,f'({chr(97+i)})',transform=ax.transAxes,va='top')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,fontsize=8)
    fig.subplots_adjust(left=.10,right=.98,bottom=.19,top=.85,wspace=.37)
    return fig,checks

def main():
    OUT.mkdir(parents=True,exist_ok=True);plt.style.use(ROOT/'src/lenscluster/style.mplstyle');plt.rcParams.update(PAPER_STYLE)
    f1,c1=pair();f2=field();f3,c3=averaging()
    assert np.allclose(corr(0),[1,1])
    with PdfPages(OUT/'covariance_intuition_all.pdf') as pdf:
        for fig,name in zip([f1,f2,f3],['pair_residual_draws','conditional_deflection_field','averaging_correlated_images']):
            fig.savefig(OUT/f'{name}.pdf');pdf.savefig(fig);plt.close(fig)
    (OUT/'checks.json').write_text(json.dumps({'pair_covariances':c1,'average_variances':c3},indent=2))
    print('Saved three individual figures and a combined PDF. Covariance positivity and zero-separation normalization passed.')
if __name__=='__main__':main()
