"""PDF-only schematics of the flux likelihood described in paper/main.tex.

Synthetic examples, not fitted cluster data. The flux likelihood implementation
was not present in this checkout when these figures were created. Magnitude
correction uses Mhat=m-2.5 log10(Y), consistent with m=M+2.5 log10(Y).
"""
from pathlib import Path
import numpy as np
from scipy.special import erf
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

OUT=Path(__file__).resolve().parents[1]/'plots/paper/flux_candidates'
BLUE,GREEN,RED,PURPLE,INK='#1674A4','#218778','#C44E35','#77529B','#252930'
COLORS=[BLUE,GREEN,PURPLE]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'mathtext.fontset':'dejavusans',
 'text.color':INK,'axes.labelcolor':INK,'axes.edgecolor':INK,'axes.grid':False,
 'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})

def normal(x,m,s):
    return np.exp(-.5*((x-m)/s)**2)/(np.sqrt(2*np.pi)*s)

def marginal(m,s):
    w=1/s**2; mean=np.sum(w*m)/w.sum()
    logl=-.5*np.sum((m-mean)**2/s**2+np.log(2*np.pi*s**2))-.5*np.log(w.sum())+.5*np.log(2*np.pi)
    return mean,logl

def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True)
    p=OUT/(name+'.pdf'); fig.savefig(p,bbox_inches='tight',pad_inches=.18);plt.close(fig);print(p)

def alignment():
    m0=26.; Y=np.array([.045,.13,.3]); errors=np.array([.14,.19,.16])
    corrected=m0+np.array([-.07,.08,-.01]); observed=corrected+2.5*np.log10(Y)
    assert np.allclose(observed-2.5*np.log10(Y),corrected)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for i,c in enumerate(COLORS):
        axes[0].errorbar(i,observed[i],yerr=[.08,.10,.09][i],fmt='o',color=c,capsize=3,label=f'Image {i+1}')
        axes[1].errorbar(i,corrected[i],yerr=errors[i],fmt='o',color=c,capsize=3)
    mean,_=marginal(corrected,errors)
    axes[1].axhline(mean,color=INK,ls='--',lw=1.2,label='Common source magnitude')
    for ax in axes:
        ax.set_xticks([0,1,2],['Image 1','Image 2','Image 3']); ax.set_xlim(-.55,2.55)
        ax.invert_yaxis()
    axes[0].set_ylabel(r'Observed magnitude, $m_i$')
    axes[1].set_ylabel(r'Corrected magnitude, $\widehat M_i$')
    axes[1].set_ylim(26.55,25.45)
    axes[1].legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,1.14),fontsize=10)
    fig.subplots_adjust(wspace=.38,bottom=.25,top=.85)
    fig.text(.5,.055,r'$\widehat M_i=m_i-2.5\log_{10}Y_i$',ha='center',fontsize=16)
    save(fig,'flux_magnitude_alignment')

def marginalization():
    x=np.linspace(24.8,27.2,2001); s=np.array([.19,.23,.21])
    examples=[np.array([25.92,26.08,26.02]),np.array([25.65,26.36,26.02])]
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    results=[]
    for ax,m in zip(axes,examples):
        prod=np.ones_like(x)
        for i,c in enumerate(COLORS):
            p=normal(x,m[i],s[i]);prod*=p
            ax.plot(x,p,color=c,lw=1.7,label=f'Image {i+1}')
        ax.fill_between(x,prod,color=INK,alpha=.12)
        ax.plot(x,prod,color=INK,lw=2,label='Product of image likelihoods')
        mean,ll=marginal(m,s);results.append(ll)
        integral=quad(lambda z: np.prod(normal(z,m,s)),24,28,epsabs=1e-11)[0]
        assert np.isclose(np.log(integral),ll,atol=1e-9)
        assert np.isclose(marginal(m+3,s)[1],ll)
        ax.set(xlim=(25.1,26.9),ylim=(0,8.5),xlabel=r'Unknown source magnitude, $M$',ylabel='Likelihood / product')
        ax.set_yticks([]);ax.set_xticks([])
    axes[0].text(.5,.9,'Consistent corrected magnitudes',transform=axes[0].transAxes,ha='center',fontsize=11)
    axes[1].text(.5,.9,'Inconsistent corrected magnitudes',transform=axes[1].transAxes,ha='center',fontsize=11)
    axes[0].text(.5,.76,'Larger integrated likelihood',transform=axes[0].transAxes,ha='center',fontsize=10,color=GREEN)
    axes[1].text(.5,.76,'Smaller integrated likelihood',transform=axes[1].transAxes,ha='center',fontsize=10,color=RED)
    assert results[0]>results[1]
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=4,frameon=False,fontsize=10)
    fig.subplots_adjust(wspace=.2,bottom=.25,top=.82)
    fig.text(.5,.055,r'$\mathcal{L}_{\rm mag}\propto\int dM\,\prod_i\mathcal{N}(\widehat M_i-M;\sigma_i^2)$',ha='center',fontsize=16)
    save(fig,'flux_likelihood_marginalization')

def propagation():
    D=np.linspace(-.5,.5,1601); s=.045; sd=.035; zeta=.055
    st=np.hypot(s,sd); k=2.5/np.log(10)
    h=s*np.sqrt(2/np.pi)*np.exp(-.5*(D/s)**2)+D*erf(D/(np.sqrt(2)*s))
    Y=np.sqrt(h*h+zeta*zeta)
    sig=k*st/np.sqrt(Y*Y+8/np.pi**2*st*st)
    phot=.10; extra=.07; sys=.18
    total=np.sqrt(sig**2+phot**2+extra**2+sys**2)
    assert np.isfinite(total).all() and np.all(Y>=zeta)
    fig,axes=plt.subplots(1,3,figsize=(13,4.5))
    axes[0].plot(D,abs(D),color=INK,lw=1.5,ls='--',label=r'Point determinant $|D|$')
    axes[0].plot(D,h,color=GREEN,lw=2,label=r'Position-averaged $h$')
    axes[0].plot(D,Y,color=PURPLE,lw=2,label=r'Finite-source effective $Y$')
    axes[0].set_ylabel('Inverse magnification')
    naive=k*st/np.maximum(abs(D),1e-10)
    axes[1].plot(D,naive,color=INK,lw=1.5,ls='--',label='First-order propagation')
    axes[1].plot(D,sig,color=RED,lw=2,label='Regularized uncertainty')
    axes[1].set(ylim=(0,1.7),ylabel='Magnification uncertainty (mag)')
    axes[2].plot(D,sig,color=RED,lw=2,label='Magnification')
    axes[2].axhline(np.hypot(phot,extra),color=BLUE,lw=1.5,label='Photometry')
    axes[2].axhline(sys,color=PURPLE,lw=1.5,ls=':',label='Additional scatter')
    axes[2].plot(D,total,color=INK,lw=2.2,label='Total')
    axes[2].set(ylim=(0,1.7),ylabel='Magnitude uncertainty (mag)')
    for ax in axes:
        ax.axvline(0,color=RED,lw=.8,alpha=.35,zorder=0)
        ax.set_xlim(-.5,.5);ax.set_xlabel(r'Determinant mean, $\nu=D$')
        ax.legend(frameon=False,loc='lower center',bbox_to_anchor=(.5,1.04),fontsize=9)
    fig.subplots_adjust(wspace=.35,bottom=.28,top=.73)
    fig.text(.5,.09,r'$Y^2=h^2+\zeta^2\qquad\qquad\sigma_{\rm mag}^2=(2.5/\ln10)^2\,s_{\rm tot}^2/(Y^2+8s_{\rm tot}^2/\pi^2)$',ha='center',fontsize=13)
    save(fig,'flux_uncertainty_mapping')

def covariances():
    # Two bands of one image: common magnification and systematic shift plus
    # independent photometry. Contours show one-standard-deviation axes.
    photo=np.diag([.08**2+.06**2,.12**2+.06**2]); shared=.22**2*np.ones((2,2)); C=photo+shared
    fig,axes=plt.subplots(1,2,figsize=(10,4.8),gridspec_kw={'width_ratios':[1,1.1]})
    ax=axes[0]
    for cov,c,label in [(photo,BLUE,'Independent photometry'),(C,INK,'Total covariance')]:
        vals,vec=np.linalg.eigh(cov);angle=np.degrees(np.arctan2(vec[1,-1],vec[0,-1]))
        ax.add_patch(Ellipse((0,0),2*np.sqrt(vals[-1]),2*np.sqrt(vals[0]),angle=angle,
                            fill=False,edgecolor=c,lw=2,label=label))
    v=.22
    ax.plot([-v,v],[-v,v],color=RED,lw=2.3,label='Shared magnification + scatter')
    ax.plot(0,0,'o',color=INK,ms=3)
    ax.set(xlim=(-.42,.42),ylim=(-.42,.42),aspect='equal',xlabel=r'Band 1 offset, $\delta m_1$',ylabel=r'Band 2 offset, $\delta m_2$')
    ax.set_xticks([]);ax.set_yticks([])
    ax.legend(frameon=False,loc='lower center',bbox_to_anchor=(.5,1.04),fontsize=9)
    full=np.zeros((6,6))
    for i,amp in enumerate([.22,.16,.28]):
        full[2*i:2*i+2,2*i:2*i+2]=photo+amp**2
    corr=full/np.sqrt(np.outer(full.diagonal(),full.diagonal()))
    assert np.all(np.linalg.eigvalsh(full)>0)
    ax=axes[1]
    im=ax.imshow(corr,cmap='Blues',vmin=0,vmax=1)
    labels=['1a','1b','2a','2b','3a','3b']
    ax.set_xticks(range(6),labels);ax.set_yticks(range(6),labels)
    ax.tick_params(length=0)
    ax.set_xlabel('Image and band');ax.set_ylabel('Image and band')
    for i in range(6):
        for j in range(6):
            if i!=j and corr[i,j]>0:
                ax.text(j,i,'shared',ha='center',va='center',fontsize=7,color='white')
    fig.colorbar(im,ax=ax,fraction=.045,pad=.05,label='Correlation')
    fig.subplots_adjust(wspace=.45,bottom=.25,top=.75)
    fig.text(.5,.07,r'$V_{ib,jc}=\delta_{ij}\left[\delta_{bc}\left[(\sigma_{ib}^{\rm phot})^2+\sigma_{\rm phot,extra}^2\right]+\sigma_{{\rm shared},i}^2\right]$',ha='center',fontsize=13)
    save(fig,'flux_multiband_covariance')

if __name__=='__main__':
    alignment();marginalization();propagation();covariances()
