"""PDF-only illustrations of latent-position marginalization.

Synthetic one-dimensional examples, not cluster fits. The first uses a
normalized symmetric fold pair, beta=x**2/2, with an equal branch mixture.
The other figures evaluate the draft's linearized Gaussian model exactly.
"""
from pathlib import Path
import sys
import numpy as np
from scipy.special import erf
from scipy.integrate import quad
from scipy.signal import fftconvolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lenscluster.plot_style import PAPER_STYLE

OUT = Path(__file__).resolve().parents[1] / 'output/pdf/likelihood_candidates'
BLUE, GREEN, RED, PURPLE, INK = '#1674A4', '#218778', '#C44E35', '#77529B', '#252930'
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
    'text.color':INK, 'axes.labelcolor':INK, 'axes.edgecolor':INK,
    'axes.spines.top':False, 'axes.spines.right':False,
    'axes.grid':False, 'pdf.fonttype':42, 'lines.linewidth':2.3})

def normal(x, mean=0, sigma=1):
    return np.exp(-.5*((x-mean)/sigma)**2)/(np.sqrt(2*np.pi)*sigma)

def folded(nu, s):
    return s*np.sqrt(2/np.pi)*np.exp(-.5*(nu/s)**2)+nu*erf(nu/(np.sqrt(2)*s))

def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / (name+'.pdf')
    fig.savefig(path, bbox_inches='tight', pad_inches=.22)
    plt.close(fig)
    print(path)

def critical(ax, height=.84):
    ax.axvline(0, ymax=height, color=RED, lw=1.2, alpha=.8, zorder=0)
    ax.text(0, .98, 'Critical curve', transform=ax.get_xaxis_transform(),
            color=RED, ha='center', va='top', fontsize=10)

@plt.rc_context(fname=ROOT / 'src/lenscluster/style.mplstyle',
                rc={**PAPER_STYLE, 'mathtext.fontset': 'dejavusans',
                    'mathtext.default': 'regular', 'axes.spines.top': False,
                    'axes.spines.right': False})
def convolution():
    x=np.linspace(-6,6,12001); dx=x[1]-x[0]
    # Normalize the fold-pair position density explicitly; this avoids treating
    # the many-to-one change of variables as a globally normalized density.
    true=np.abs(x)*normal(x*x/2, .55, .55)
    true/=np.trapezoid(true,x)
    kernel=normal(x,0,.42)
    measured=fftconvolve(true,kernel,mode='same')*dx
    assert abs(np.trapezoid(measured,x)-1)<1e-8
    assert true[len(x)//2]==0 and measured[len(x)//2]>0
    product=true*kernel
    v=measured[len(x)//2]
    assert np.isclose(np.trapezoid(product,x),v,rtol=1e-10)
    fig,ax=plt.subplots(figsize=(5.4,3.5))
    ax.plot(x,true,color=PURPLE,ls='--',lw=1.7,
            label='Model latent-position density')
    ax.plot(x,kernel,color=GREEN,lw=1.8,
            label='Measurement error')
    ax.plot(x,measured,color=BLUE,lw=2.2,
            label='Marginalized likelihood')
    ax.set(xlim=(-2.7,2.7),ylim=(0,1.12),
           xlabel='Image-plane position across the critical curve',
           ylabel='Probability density')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axvline(0, ymax=.88, color=RED, lw=1.2, alpha=.7, zorder=0)
    ax.text(0, .98, 'Critical curve', transform=ax.get_xaxis_transform(),
            color=RED, ha='center', va='top', fontsize=9)
    ax.plot(0,0,'o',mfc='white',mec=PURPLE,mew=1.3,ms=5,clip_on=False,zorder=5,
            label=r'Observed position, $\theta_i^{\rm meas}$')
    ax.plot(0,v,'D',mfc='white',mec=BLUE,mew=1.5,ms=6,zorder=6,
            label='Likelihood at the\nobserved position')
    handles,labels=ax.get_legend_handles_labels()
    curve_order=[0,2,1]
    fig.legend([handles[j] for j in curve_order],[labels[j] for j in curve_order],
               loc='upper left',bbox_to_anchor=(.105,.99),frameon=False,
               handlelength=2.4,handletextpad=.6)
    fig.legend(handles[3:5],labels[3:5],loc='upper right',bbox_to_anchor=(.985,.99),
               frameon=False,handletextpad=.6,markerscale=1)
    fig.subplots_adjust(left=.105,right=.98,bottom=.21,top=.82)
    terms = [
        (r'$p(\theta^{\rm meas})$', BLUE),
        (r'$=\int$', INK),
        (r'$p(\theta^{\rm meas}\mid\theta)$', GREEN),
        (r'$p(\theta)$', PURPLE),
        (r'$d\theta$', INK),
    ]
    equation = HPacker(children=[TextArea(text, textprops={'color': color, 'fontsize': 12})
                                for text, color in terms], align='baseline', pad=0, sep=4)
    fig.add_artist(AnchoredOffsetbox(loc='center', child=equation, frameon=False,
                                    bbox_to_anchor=(.535, .10), bbox_transform=fig.transFigure,
                                    pad=0, borderpad=0))
    paper_path = ROOT / 'plots/paper/likelihood_marginalization.pdf'
    paper_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paper_path, bbox_inches='tight', pad_inches=.03)
    OUT.mkdir(parents=True, exist_ok=True)
    # Keep the existing review filename identical to the publication version.
    (OUT / 'true_to_measured_position.pdf').write_bytes(paper_path.read_bytes())
    plt.close(fig)
    print(paper_path)

def conditioning():
    sm,a,cs,F,D,g=.8,.7,.65,.7,.15,1.
    C=cs**2+(sm*a)**2
    mean=-sm**2*a*F/C
    s=sm*cs/np.sqrt(C)
    nu=D+g*mean
    u=np.linspace(-3.3,3.3,1801)
    prior=normal(u,0,sm)
    source=normal(F+a*u,0,cs)
    cond=normal(u,mean,s)
    assert np.allclose(prior*source,normal(F,0,np.sqrt(C))*cond)
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.8))
    ax=axes[0]
    ax.plot(u,prior,color=BLUE,label=r'Measurement alone: $p(u)$')
    ax.plot(u,source/source.max()*prior.max(),color=PURPLE,ls='--',
            label='Source consistency (rescaled)')
    ax.plot(u,cond,color=GREEN,label=r'Conditioned displacement: $p(u\mid F)$')
    ax.fill_between(u,cond,color=GREEN,alpha=.1)
    ax.set(xlim=(-2.6,2.6),ylim=(0,1.04),xlabel=r'True minus measured position, $u$',ylabel='Probability density')
    ax.axvline(0,ymax=.67,color=BLUE,lw=1,ls=':')
    ax.annotate('Residual shifts and narrows\nthe allowed displacement',xy=(mean,.48),xytext=(.28,.74),
                fontsize=10,color=GREEN,arrowprops={'arrowstyle':'-','color':GREEN})
    ax.legend(frameon=False,loc='upper left',bbox_to_anchor=(0,1.31),fontsize=10)
    ax=axes[1]
    X=np.linspace(-3,3,1801); p=normal(X,nu,s)
    ax.plot(X,p,color=GREEN,label=r'Conditional determinant: $p(X\mid F)$')
    ax.fill_between(X,0,np.abs(X)*p,where=X<0,color=PURPLE,alpha=.24,label='Negative Parity contribution')
    ax.fill_between(X,0,np.abs(X)*p,where=X>=0,color=RED,alpha=.23,label='Positive Parity contribution')
    ax.axvline(0,ymax=.74,color=INK,lw=1,ls=':')
    ax.set(xlim=(-2.6,2.6),ylim=(0,1.04),xlabel=r'Local determinant, $X=D+g\,u$',ylabel='Density / weighted density')
    ax.text(.55,.55,'Both shaded areas\ncontribute to $h$',fontsize=11,color=INK)
    ax.legend(frameon=False,loc='upper left',bbox_to_anchor=(0,1.31),fontsize=10)
    numerical=quad(lambda z: abs(z)*normal(z,nu,s),-np.inf,np.inf)[0]
    assert abs(numerical-folded(nu,s))<1e-9
    fig.subplots_adjust(wspace=.3,bottom=.27,top=.73)
    fig.text(.5,.07,r'$\mathcal{L}_{\rm pos}=\mathcal{N}(F;C)\,h,\qquad h=\int |X|\,p(X\mid F)\,dX$',ha='center',fontsize=16)
    save(fig,'conditional_jacobian_average')

def smoothing():
    D=np.linspace(-2.5,2.5,1001)
    sm,a,cs=.8,.7,.65
    C=cs**2+(sm*a)**2
    s=sm*cs/np.sqrt(C)
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.7))
    for ax in axes:
        ax.set(xlim=(-2.5,2.5),ylim=(0,3.45),xlabel=r'Determinant at the measured position, $D$')
        ax.axvline(0,ymax=.82,color=RED,alpha=.6,lw=1.1)
        ax.plot(D,abs(D),color=INK,ls='--',lw=1.5,label=r'Point evaluation: $|D|$')
    for F,color,label in [(0,GREEN,r'$F=0$'),(.9,PURPLE,r'$F>0$'),(-.9,BLUE,r'$F<0$')]:
        shift=sm**2*a*F/C
        h=folded(D-shift,s)
        axes[0].plot(D,h,color=color,label=label)
        axes[1].plot(D,np.log(h),color=color,label=label)
        assert np.all(h>0)
    # Replace the second panel's reference by its logarithm.
    axes[1].lines[1].remove()
    nonzero=np.abs(D)>1e-10
    axes[1].plot(D,np.where(nonzero,np.log(np.maximum(abs(D),1e-20)),np.nan),
                 color=INK,ls='--',lw=1.5,label=r'Point evaluation: $\ln|D|$')
    axes[1].set(ylim=(-3,1.5),ylabel=r'Log Jacobian factor, $\ln h$')
    axes[0].set_ylabel(r'Jacobian factor, $h$')
    axes[0].legend(frameon=False,ncol=2,loc='upper center',fontsize=10)
    axes[1].annotate('Point evaluation\ndiverges at $D=0$',xy=(.04,-2.8),xytext=(.65,-2.1),
                     fontsize=10,arrowprops={'arrowstyle':'-','color':INK})
    axes[0].text(0,2.85,'Critical curve',color=RED,ha='center',fontsize=10)
    axes[1].text(0,1.2,'Critical curve',color=RED,ha='center',fontsize=10)
    fig.subplots_adjust(wspace=.28,bottom=.27,top=.94)
    fig.text(.5,.08,'The residual shifts the conditional mean; the smoothed factor stays finite at $D=0$.',ha='center',fontsize=11)
    save(fig,'residual_dependent_smoothing')



@plt.rc_context(fname=ROOT / 'src/lenscluster/style.mplstyle',
                rc={**PAPER_STYLE, 'mathtext.fontset': 'dejavusans',
                    'mathtext.default': 'regular', 'axes.spines.top': False,
                    'axes.spines.right': False})
def determinant_comparison():
    """Zero-residual local example: nu=D, with fixed conditional width s.

The determinant is linear across a regular critical curve. The displayed
factor is its absolute value, as required by the positional likelihood.
"""
    D=np.linspace(-1.8,1.8,1201)
    s=.65
    h=folded(D,s)
    assert np.all(h >= np.abs(D)-1e-12)
    assert np.isclose(folded(0,s),s*np.sqrt(2/np.pi))
    errors=[]
    for mean in [-1.8, -.65, 0., .65, 1.8]:
        integrand=lambda X: abs(X)*normal(X, mean, s)
        reference=sum(quad(integrand, lo, hi, epsabs=1e-12, epsrel=1e-12)[0]
                      for lo,hi in [(-np.inf,0.),(0.,np.inf)])
        errors.append(abs(reference-folded(mean,s)))
    assert max(errors)<1e-12
    fig,ax=plt.subplots(figsize=(7.1,3.9))
    ax.plot(D,np.abs(D),color=PURPLE,ls='--',lw=1.7,
            label=r'Point evaluation: $|D|$')
    ax.plot(D,h,color=BLUE,lw=2.2,
            label=r'Smoothed Jacobian factor: $h$')
    ax.plot(0,0,'o',mfc='white',mec=PURPLE,mew=1.3,ms=5,clip_on=False,zorder=5)
    ax.plot(0,folded(0,s),'D',mfc='white',mec=BLUE,mew=1.5,ms=6,
            zorder=6,label='Smoothed value at the critical curve')
    ax.set(xlim=(-1.8,1.8),ylim=(0,2.4),
           xlabel=r'Jacobian determinant at the measured position, $D$',
           ylabel='Jacobian factor')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axvline(0,ymax=.88,color=RED,lw=1.2,alpha=.7,zorder=0)
    ax.text(0,.98,'Critical curve',transform=ax.get_xaxis_transform(),
            color=RED,ha='center',va='top',fontsize=9)
    ax.text(-1.1,2.08,'Negative parity',ha='center',fontsize=9,color=INK)
    ax.text(1.1,2.08,'Positive parity',ha='center',fontsize=9,color=INK)
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles[:2],labels[:2],loc='upper left',bbox_to_anchor=(.085,.99),
               frameon=False,fontsize=9,handlelength=2.4,handletextpad=.6)
    fig.legend(handles[2:],labels[2:],loc='upper right',bbox_to_anchor=(.985,.99),
               frameon=False,fontsize=9,handletextpad=.6,markerscale=1)
    fig.subplots_adjust(left=.085,right=.985,bottom=.29,top=.78)
    terms=[(r'$h$',BLUE),(r'$=\int$',INK),(r'$|X|$',PURPLE),
           (r'$p(X\mid\mathbf{F})$',GREEN),(r'$dX$',INK)]
    equation=HPacker(children=[TextArea(t,textprops={'color':c,'fontsize':12}) for t,c in terms],
                     align='baseline',pad=0,sep=4)
    fig.add_artist(AnchoredOffsetbox(loc='center',child=equation,frameon=False,
                    bbox_to_anchor=(.535,.105),bbox_transform=fig.transFigure,pad=0,borderpad=0))
    fig.text(.535,.025,r'Zero-residual illustration: $\nu=D$, fixed conditional width $s$',
             ha='center',fontsize=8.5)
    path=ROOT/'plots/paper/jacobian_marginalization.pdf'
    path.parent.mkdir(parents=True,exist_ok=True)
    with plt.rc_context({'savefig.bbox':None}):
        fig.savefig(path,bbox_inches=None,pad_inches=0)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'jacobian_determinant_average.pdf').write_bytes(path.read_bytes())
    plt.close(fig)
    print(f'{path}\nMaximum absolute quadrature error: {max(errors):.3g}')


if __name__=='__main__':
    convolution()
    conditioning()
    smoothing()
    determinant_comparison()
