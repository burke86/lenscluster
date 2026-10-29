"""Synthetic illustrations of E[|X| | F] for a conditional Gaussian determinant.

Run in ggsl, hdm or lenstronomy. Uses NumPy, SciPy and Matplotlib.
The conditional example uses one standardized residual component, not a cluster fit.
"""
from pathlib import Path
import json
import numpy as np
from scipy.special import erf
from scipy.integrate import quad
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[1]/"output/pdf/jacobian_smoothing_candidates"
BLUE,GREEN,RED,PURPLE,INK="#1674A4","#218778","#C44E35","#77529B","#252930"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,
                    "mathtext.fontset":"stix","text.color":INK,
                    "axes.labelcolor":INK,"axes.edgecolor":INK,
                    "axes.spines.top":False,"axes.spines.right":False,
                    "axes.grid":False,"pdf.fonttype":42})


def h(nu,s):
    return s*np.sqrt(2/np.pi)*np.exp(-.5*(nu/s)**2)+nu*erf(nu/(np.sqrt(2)*s))


def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/(name+".pdf"),bbox_inches="tight",pad_inches=.15)
    fig.savefig(OUT/(name+".png"),bbox_inches="tight",pad_inches=.15,dpi=180)
    plt.close(fig)


def parity_average():
    x=np.linspace(-4,4.5,1200)
    nu=.5
    density=norm.pdf(x,nu,1)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    for ax,values in zip(axes,[density,abs(x)*density]):
        ax.plot(x,values,color=INK,lw=1.8)
        ax.fill_between(x,0,values,where=x<0,color=RED,alpha=.19)
        ax.fill_between(x,0,values,where=x>=0,color=BLUE,alpha=.19)
        ax.axvline(0,color=INK,alpha=.3,lw=.9)
        ax.set_xlim(-4,4.5)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r"$X/s$")
    axes[0].set_ylabel(r"$s\,p(X\mid\mathbf{F})$")
    axes[1].set_ylabel(r"$|X|\,p(X\mid\mathbf{F})$")
    axes[0].text(.05,.91,r"$X<0$",color=RED,transform=axes[0].transAxes)
    axes[0].text(.78,.91,r"$X>0$",color=BLUE,transform=axes[0].transAxes)
    axes[0].text(.05,.78,r"$\nu/s=0.5$",transform=axes[0].transAxes)
    axes[0].axvline(nu,color=PURPLE,lw=1.2,ls=":")
    axes[1].set_ylim(0,.55)
    axes[1].text(.08,.94,"Both shaded areas contribute",transform=axes[1].transAxes)
    axes[1].text(.08,.83,r"$h/s=\int |X|\,p(X\mid\mathbf{F})\,dX/s$",
                 transform=axes[1].transAxes,fontsize=15)
    fig.subplots_adjust(bottom=.22,wspace=.3)
    fig.text(.5,.015,r"$h=\mathbb{E}[|X|\mid\mathbf{F}]$"
             r"$\qquad\neq\qquad |\mathbb{E}[X\mid\mathbf{F}]|$",
             ha="center",fontsize=20)
    save(fig,"jacobian_parity_average")


def response():
    z=np.linspace(-4,4,1001)
    smooth=h(z,1)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    axes[0].plot(z,abs(z),color=INK,ls="--",lw=1.4,label=r"$|\nu|/s$")
    axes[0].plot(z,smooth,color=GREEN,lw=2,label=r"$h/s$")
    axes[0].plot(0,np.sqrt(2/np.pi),"o",mfc="white",mec=PURPLE,mew=1.6,ms=7)
    axes[0].annotate(r"$\sqrt{2/\pi}$",xy=(0,np.sqrt(2/np.pi)),xytext=(1.15,.6),
                     color=PURPLE,fontsize=16,arrowprops=dict(arrowstyle="-",color=PURPLE))
    axes[0].set(xlabel=r"$\nu/s$",ylabel="Normalized Jacobian factor",ylim=(0,4.3))
    axes[0].legend(frameon=False,loc="upper center")
    for sign in [-1,1]:
        zz=np.geomspace(.001,4,700)
        axes[1].plot(sign*zz,np.log(zz),color=INK,ls="--",lw=1.4,
                     label=r"$\ln(|\nu|/s)$" if sign==1 else None)
    axes[1].plot(z,np.log(smooth),color=GREEN,lw=2,label=r"$\ln(h/s)$")
    axes[1].set(xlabel=r"$\nu/s$",ylabel="Logarithmic normalization",ylim=(-4,1.6))
    axes[1].legend(frameon=False,loc="lower right")
    for ax in axes:
        ax.set_xlim(-4,4)
    fig.subplots_adjust(bottom=.22,wspace=.3)
    fig.text(.5,.015,r"$h=s\sqrt{2/\pi}\,e^{-\nu^2/(2s^2)}"
             r"+\nu\,\mathrm{erf}\!\left(\nu/(\sqrt{2}\,s)\right)$",
             fontsize=19,ha="center")
    save(fig,"jacobian_smoothed_factor")


def conditioning():
    rho=.7
    f=.9
    nu=rho*f
    s=np.sqrt(1-rho*rho)
    u=np.linspace(-3,3,300)
    F,X=np.meshgrid(u,u)
    q=(F*F-2*rho*F*X+X*X)/(1-rho*rho)
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    axes[0].contourf(F,X,np.exp(-q/2),levels=[np.exp(-4.5),np.exp(-2),np.exp(-.5),1],
                     colors=[BLUE,BLUE,BLUE],alpha=.12)
    axes[0].contour(F,X,q,levels=[1,4,9],colors=BLUE,linewidths=[1.6,1,.7])
    axes[0].axvline(f,color=PURPLE,lw=1.5)
    axes[0].plot(f,nu,"o",color=PURPLE,ms=6)
    axes[0].text(f+.15,-2.65,r"$F=F_{\rm obs}$",color=PURPLE)
    axes[0].set(xlim=(-3,3),ylim=(-3,3),xlabel="Standardized positional residual",
                ylabel="Standardized determinant")
    axes[0].set_aspect("equal")
    axes[1].plot(u,norm.pdf(u),color=BLUE,lw=1.8,label="Before conditioning")
    axes[1].plot(u,norm.pdf(u,nu,s),color=PURPLE,lw=1.8,label=r"Conditioned on $F_{\rm obs}$")
    axes[1].fill_between(u,0,norm.pdf(u,nu,s),color=PURPLE,alpha=.08)
    axes[1].set(xlim=(-3,3),ylim=(0,.65),xlabel="Standardized determinant",ylabel="Probability density")
    axes[1].legend(frameon=False,fontsize=10,loc="upper left")
    axes[1].text(.08,.72,"Shifted mean,\nsmaller variance",transform=axes[1].transAxes,
                 color=PURPLE,fontsize=11)
    fig.subplots_adjust(bottom=.25,wspace=.28)
    fig.text(.5,.065,r"$\nu=D+\mathbf{c}^{\mathsf{T}}\mathbf{C}^{-1}\mathbf{F}"
             r"\qquad s^2=V_X-\mathbf{c}^{\mathsf{T}}\mathbf{C}^{-1}\mathbf{c}$",
             fontsize=19,ha="center")
    fig.text(.5,.005,r"One-residual Gaussian example; $V_X$ denotes the determinant variance before conditioning.",
             fontsize=10,ha="center")
    save(fig,"jacobian_conditioning")


def main():
    errors=[]
    for nu in [-3,-.5,0,.5,3]:
        numeric=quad(lambda x: abs(x)*norm.pdf(x,nu,1),-np.inf,np.inf)[0]
        errors.append(abs(numeric-h(nu,1)))
    assert max(errors)<1e-9
    z=np.linspace(-6,6,1001)
    assert np.all(h(z,1)>=abs(z)-1e-12)
    assert np.allclose(h(z,1),h(-z,1))
    parity_average()
    response()
    conditioning()
    (OUT/"checks.json").write_text(json.dumps(dict(
        maximum_quadrature_error=max(errors),parity_symmetric=True,
        conditional_example=dict(correlation=.7,standardized_residual=.9),
        provenance="Synthetic Gaussian examples; not cluster-fit results."),indent=2)+"\n")
    print("Three figures generated; Gaussian quadrature, parity and Jensen checks passed.")


if __name__=="__main__":
    main()
