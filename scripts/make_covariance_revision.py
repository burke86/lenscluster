"""Synthetic illustrations of paper/main.tex's revised joint covariance.

No fit products are read. Outputs remain separate from covariance_mapping.pdf.
Run with the authorized lenscluster or lenstronomy Python interpreter.
"""
from pathlib import Path
import sys, json
import numpy as np
from scipy.special import jv
from numpy.polynomial.legendre import leggauss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from cmcrameri import cm
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from lenscluster.plot_style import PAPER_STYLE
OUT=ROOT/'plots/paper/covariance_revision'
COLORS=['#0072B2','#D55E00','#009E73','#222222']
LABELS=['Measurement','Deflection GRF','Image-plane GP','Total']
STYLES=[':', '--', '-.', '-']
SM,SA,SH,ELL,KNEE=.03,.04,.06,1.5,6.

def correlations(r,n=1400):
    """Normalized potential-gradient covariance, P_kappa dlnk weight."""
    nodes,weights=leggauss(n)
    lo,hi=np.log(2*np.pi/100),np.log(2*np.pi/.2)
    k=np.exp((hi+lo)/2+(hi-lo)/2*nodes)
    weight=weights/(1+(k/(2*np.pi/KNEE))**4)
    kr=np.asarray(r)[...,None]*k
    j0,j2=jv(0,kr),jv(2,kr)
    return ((j0-j2)@weight/weight.sum(),(j0+j2)@weight/weight.sum())

def covariance(xy,A,w):
    n=len(xy); meas=np.zeros((n,2,n,2)); mass=meas.copy(); img=meas.copy()
    for i in range(n):
        meas[i,:,i,:]=SM**2*A[i]@A[i].T
        for j in range(n):
            d=xy[i]-xy[j]; r=np.linalg.norm(d)
            cl,ct=correlations(r,n=800)
            direction=d/r if r else np.zeros(2)
            mass[i,:,j,:]=w[i]*w[j]*SA**2*(ct*np.eye(2)+(cl-ct)*np.outer(direction,direction))
            img[i,:,j,:]=SH**2*np.exp(-r*r/(2*ELL**2))*A[i]@A[j].T
    parts=[c.reshape(2*n,2*n) for c in (meas,mass,img)]
    return parts+[sum(parts)]

def legend(fig):
    fig.legend([Line2D([],[],color=c,ls=s,lw=1.5) for c,s in zip(COLORS,STYLES)],LABELS,
               loc='upper center',bbox_to_anchor=(.5,1.01),ncol=4,frameon=False)

def letter(ax,t): ax.text(.04,.95,t,transform=ax.transAxes,va='top',fontsize=10)

def mapping(*, sm=SM, sa=SA, sh=SH, lt=.12, lr=.7, angle_deg=25., floor=0.):
    fig,axes=plt.subplots(2,2,figsize=(7.1,6.3))
    axes=axes.ravel()
    a=np.deg2rad(angle_deg); R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    A=R@np.diag([lt,lr])@R.T
    sa=np.hypot(sa,floor)  # diagonal deflection plus independent source floor
    src=[sm**2*A@A.T,sa**2*np.eye(2),sh**2*A@A.T]; src.append(sum(src))
    inv=np.linalg.inv(A); image=[inv@c@inv.T for c in src]
    for ax,parts,lim,plane in zip(axes[:2],[image,src],[1.2*np.sqrt(np.linalg.eigvalsh(image[-1]).max()),1.2*np.sqrt(np.linalg.eigvalsh(src[-1]).max())],[r'\theta',r'\beta']):
        for c,color,ls in zip(parts,COLORS,STYLES):
            eig,vec=np.linalg.eigh(c); angle=np.rad2deg(np.arctan2(vec[1,1],vec[0,1]))
            ax.add_patch(Ellipse((0,0),2*np.sqrt(eig[1]),2*np.sqrt(eig[0]),angle=angle,fill=False,ec=color,ls=ls,lw=1.35))
        ax.plot(0,0,'+',color='black',ms=5,mew=.8)
        ax.set(xlim=(-lim,lim),ylim=(-lim,lim),aspect='equal',xlabel=rf'$\Delta {plane}_x$ [arcsec]',ylabel=rf'$\Delta {plane}_y$ [arcsec]')
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    lam=np.linspace(0,1,400)
    curves=[sm*lam,np.full_like(lam,sa),sh*lam,np.sqrt((sm**2+sh**2)*lam**2+sa**2)]
    for y,c,ls in zip(curves,COLORS,STYLES):axes[2].plot(lam,y,color=c,ls=ls)
    axes[2].plot(lt,np.sqrt((sm**2+sh**2)*lt**2+sa**2),'o',mfc='white',mec='black',ms=5)
    axes[2].set(xlabel=r'$|\lambda_{\rm t}|$',ylabel=r'$\sigma_{\rm src,t}$ [arcsec]',xlim=(0,1),ylim=(0,1.15*np.sqrt(sm**2+sh**2+sa**2)))
    image_lam=np.geomspace(.01,1,400)
    image_curves=[np.full_like(image_lam,sm),sa/image_lam,np.full_like(image_lam,sh),
                  np.sqrt(sm**2+sh**2+(sa/image_lam)**2)]
    for y,c,ls in zip(image_curves,COLORS,STYLES):
        axes[3].loglog(image_lam,y,color=c,ls=ls)
    axes[3].plot(lt,np.sqrt(sm**2+sh**2+(sa/lt)**2),'o',mfc='white',mec='black',ms=5)
    axes[3].set(xlabel=r'$|\lambda_{\rm t}|$',ylabel=r'$\sigma_{\rm img,t}$ [arcsec]',
                xlim=(.01,1),ylim=(.7*min(sm,sh,sa),1.2*np.sqrt(sm**2+sh**2+(sa/.01)**2)))
    # The inverse-Jacobian illustration is defined only for nonzero eigenvalues.
    source_curves=[sm*image_lam,np.full_like(image_lam,sa),sh*image_lam,
                   np.sqrt((sm**2+sh**2)*image_lam**2+sa**2)]
    np.testing.assert_allclose(np.array(image_curves)*image_lam,np.array(source_curves),rtol=1e-14)
    for ax,t in zip(axes,['(a)','(b)','(c)','(d)']):letter(ax,t)
    if floor:
        fig.legend([Line2D([],[],color=c,ls=t,lw=1.5) for c,t in zip(COLORS,STYLES)],
                   ["Measurement", "Deflection + floor", "Image-plane GP", "Total"],
                   loc="upper center",bbox_to_anchor=(.5,1.01),ncol=4,frameon=False)
    else:
        legend(fig)
    fig.subplots_adjust(left=.12,right=.975,bottom=.09,top=.92,wspace=.36,hspace=.38)
    return fig

def blocks():
    # Two fold pairs; t is their separation direction, with opposite tangential parity.
    xy=np.array([[-.3,0],[.3,0],[2.2,.6],[2.8,.6]])
    A=np.array([np.diag([a,.7]) for a in [-.06,.06,-.06,.06]])
    C=covariance(xy,A,np.ones(4)); vmax=max(np.max(abs(c)) for c in C)/SA**2
    fig,axes=plt.subplots(2,4,figsize=(10.6,5.9))
    ticks=[r'$1_t$',r'$1_r$',r'$2_t$',r'$2_r$',r'$3_t$',r'$3_r$',r'$4_t$',r'$4_r$']
    names=['Measurement','Deflection GRF','Image-plane GP','Total']
    for row in range(2):
        for i,(ax,c) in enumerate(zip(axes[row],C)):
            sd=np.sqrt(np.diag(c))
            values=c/SA**2 if row==0 else c/np.outer(sd,sd)
            scale=vmax if row==0 else 1
            im=ax.imshow(values,cmap=cm.vik,vmin=-scale,vmax=scale,interpolation='nearest')
            ax.set_xticks(range(8),ticks,fontsize=7);ax.set_yticks(range(8),ticks,fontsize=7)
            ax.tick_params(which='minor',bottom=False,top=False,left=False,right=False)
            ax.set_xlabel(f'({chr(97+4*row+i)}) '+names[i])
        cb=fig.colorbar(im,cax=fig.add_axes([.92,.57 if row==0 else .12,.013,.31]))
        cb.set_label(r'$C_{ia,jb}/\sigma_\alpha^2$' if row==0 else r'$C_{ia,jb}/\sqrt{C_{ia,ia}C_{jb,jb}}$')
    fig.subplots_adjust(left=.045,right=.89,bottom=.09,top=.99,wspace=.27,hspace=.40)
    return fig,C

def fold():
    r=np.geomspace(.001,20,450);cl,ct=correlations(r);rho=np.exp(-r*r/(2*ELL**2));a=r/10
    common=[.5*SM**2*a*a,.5*SA**2*(1+cl),.5*SH**2*a*a*(1-rho)]
    relative=[2*SM**2*a*a,2*SA**2*(1-cl),2*SH**2*a*a*(1+rho)]
    common.append(sum(common));relative.append(sum(relative))
    fig,axes=plt.subplots(1,3,figsize=(10.2,3.25))
    axes[0].semilogx(r,cl,color=COLORS[1],label=r'$c_{\rm L}$')
    axes[0].semilogx(r,ct,color=COLORS[1],ls='--',label=r'$c_{\rm T}$')
    axes[0].semilogx(r,rho,color=COLORS[2],label=r'$\rho_{\rm img}$')
    axes[0].legend(loc='lower left');axes[0].set(xlim=(.001,20),ylim=(-.25,1.05),xlabel=r'Separation $r$ [arcsec]',ylabel='Correlation coefficient')
    for ax,curves,yl in zip(axes[1:],[common,relative],[r'$\mathrm{SD}[(F_{1,t}+F_{2,t})/2]$ [arcsec]',r'$\mathrm{SD}[F_{1,t}-F_{2,t}]$ [arcsec]']):
        for y,c,ls in zip(curves,COLORS,STYLES):ax.plot(r,np.sqrt(np.maximum(y,0)),c=c,ls=ls)
        ax.set(xlim=(0,3),ylim=(0,.065),xlabel=r'Separation $r$ [arcsec]',ylabel=yl)
    for ax,t in zip(axes,['(a)','(b)','(c)']):letter(ax,t)
    legend(fig);fig.subplots_adjust(left=.075,right=.985,bottom=.20,top=.85,wspace=.46)
    return fig

def validate(C):
    rs=np.r_[0,np.geomspace(.001,20,80)]; c1=correlations(rs,1000);c2=correlations(rs,1800)
    error=max(float(np.max(abs(a-b))) for a,b in zip(c1,c2));assert error<1e-8,error
    assert np.allclose([c1[0][0],c1[1][0]],1)
    eig=[np.linalg.eigvalsh(c) for c in C]
    for c,e in zip(C,eig):assert np.allclose(c,c.T) and e.min()>-1e-12
    # Test the common/difference formulas by direct projection of the 4x4 covariance.
    maxerr=0
    for r in [.0,.01,.2,1.,3.]:
        a=r/10;A=np.array([np.diag([-a,.7]),np.diag([a,.7])]);parts=covariance(np.array([[-r/2,0],[r/2,0]]),A,np.ones(2))
        cl,_=correlations(r,800);rho=np.exp(-r*r/(2*ELL**2))
        expected=[[.5*SM**2*a*a,.5*SA**2*(1+cl),.5*SH**2*a*a*(1-rho)],[2*SM**2*a*a,2*SA**2*(1-cl),2*SH**2*a*a*(1+rho)]]
        for v,vals in zip([np.array([.5,0,.5,0]),np.array([1,0,-1,0])],expected):
            for c,val in zip(parts,vals):maxerr=max(maxerr,abs(float(v@c@v)-val))
    assert maxerr<1e-12
    return {'quadrature_max_abs_error':error,'fold_projection_max_abs_error':maxerr,'matrix_min_eigenvalues':[float(e.min()) for e in eig],
            'illustrative_parameters':{'sigma_meas':SM,'sigma_alpha':SA,'sigma_sh':SH,'ell':ELL,'lambda_b':KNEE,'gamma_kappa':4,'w':1,'wavelength_band':[.2,100]},'source_floor':0}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.style.use(ROOT/'src/lenscluster/style.mplstyle');plt.rcParams.update(PAPER_STYLE)
    f1=mapping();f2,C=blocks();f3=fold();checks=validate(C)
    with PdfPages(OUT/'covariance_alternatives.pdf') as combined:
        for fig,name in zip([f1,f2,f3],['covariance_local_mapping','covariance_joint_blocks','covariance_fold_modes']):
            fig.savefig(OUT/f'{name}.pdf');combined.savefig(fig);plt.close(fig)
    (OUT/'validation.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
