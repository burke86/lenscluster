"""PDF-only meeting illustration of Jacobian smoothing near a regular critical curve.

Synthetic local example: determinant linear in distance and Gaussian uncertainty
across the curve, with zero conditional mean shift. Not a full lens likelihood.
"""
from pathlib import Path
import numpy as np
from scipy.special import erf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap

OUT=Path(__file__).resolve().parents[1]/"output/pdf/jacobian_smoothing_candidates"
BLUE,GREEN,RED,PURPLE,INK="#1674A4","#218778","#C44E35","#77529B","#252930"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":12,
                    "text.color":INK,"axes.labelcolor":INK,"axes.edgecolor":INK,
                    "axes.spines.top":False,"axes.spines.right":False,
                    "axes.grid":False,"pdf.fonttype":42})

def main():
    fig,axes=plt.subplots(1,2,figsize=(12,5.5),gridspec_kw={"width_ratios":[1,1.2]})
    left,right=axes
    left.set(xlim=(-2.2,2.2),ylim=(-1.9,2.2),aspect="equal")
    left.axis("off")
    # One blue Gaussian represents positional uncertainty; white parity regions
    # avoid suggesting a second source of uncertainty.
    gx=np.linspace(-2.2,2.2,350)
    gy=np.linspace(-1.6,1.6,260)
    xx,yy=np.meshgrid(gx,gy)
    density=np.exp(-.5*((xx/.8)**2+(yy/.52)**2))
    cmap=LinearSegmentedColormap.from_list("position_uncertainty",
                                          [(1,1,1,0),(.086,.455,.643,.32)])
    left.imshow(density,extent=(-2.2,2.2,-1.6,1.6),origin="lower",
                cmap=cmap,vmin=0,vmax=1,interpolation="bilinear",zorder=0)
    left.plot([0,0],[-1.25,1.6],color=RED,lw=1.8,zorder=2)
    left.text(0,1.85,"Critical curve",color=RED,ha="center",fontsize=15)
    left.add_patch(Ellipse((0,0),1.6,1.04,facecolor="none",
                           edgecolor=BLUE,lw=1.25,ls=(0,(4,3)),zorder=3))
    left.plot(0,0,"o",color=BLUE,ms=8,zorder=5)
    left.annotate("Measured image\nposition",xy=(0,0),xytext=(-2,1.05),
                   color=BLUE,fontsize=12,linespacing=1.3,
                   arrowprops=dict(arrowstyle="-",color=BLUE,shrinkB=7))
    left.annotate("Position uncertainty",xy=(.7,-.25),
                   xytext=(.32,-1.08),color=BLUE,fontsize=12,
                   arrowprops=dict(arrowstyle="-",color=BLUE,shrinkB=4))
    left.text(0,-1.48,"Possible positions on both sides",ha="center",
              color=INK,fontsize=10.5)
    left.text(-1.2,-1.85,"Negative Parity",ha="center",fontsize=11,color=INK)
    left.text(1.2,-1.85,"Positive Parity",ha="center",fontsize=11,color=INK)
    x=np.linspace(-3,3,1001)
    smooth=np.sqrt(2/np.pi)*np.exp(-x*x/2)+x*erf(x/np.sqrt(2))
    assert np.all(smooth>=abs(x)-1e-12)
    assert abs(smooth[500]-np.sqrt(2/np.pi))<1e-12
    right.plot([0,0],[0,3.05],color=RED,alpha=.6,lw=1,zorder=0)
    right.plot(x,abs(x),color=INK,lw=1.5,ls="--",label="Exact position")
    right.plot(x,smooth,color=GREEN,lw=2.5,label="Averaged over uncertainty")
    right.plot(0,smooth[500],"o",mfc="white",mec=GREEN,mew=2,ms=8,zorder=5)
    right.plot(0,0,"o",color=INK,ms=5,clip_on=False,zorder=5)
    right.annotate("Finite weight",xy=(0,smooth[500]),xytext=(1.0,.62),
                   color=GREEN,fontsize=13,arrowprops=dict(arrowstyle="-",color=GREEN))
    right.annotate("Zero weight",xy=(0,0),xytext=(-2.45,.35),
                   color=INK,fontsize=12,arrowprops=dict(arrowstyle="-",color=INK))
    right.set(xlim=(-3,3),ylim=(0,3.6),
              ylabel="Jacobian weight (relative units)",
              xlabel="Position across the critical curve")
    right.set_xticks([-2,0,2],["Negative Parity","Critical curve","Positive Parity"])
    right.get_xticklabels()[1].set_color(RED)
    right.set_yticks([0,1,2,3])
    right.legend(frameon=False,loc="upper center",fontsize=11)
    fig.subplots_adjust(left=.035,right=.98,top=.94,bottom=.23,wspace=.3)
    fig.text(.5,.065,"Averaging over positional uncertainty keeps the Jacobian factor finite at the critical curve.",
             ha="center",fontsize=12)
    OUT.mkdir(parents=True,exist_ok=True)
    output=OUT/"jacobian_smoothing_meeting.pdf"
    fig.savefig(output,bbox_inches="tight",pad_inches=.18)
    plt.close(fig)
    print(output)

if __name__=="__main__":
    main()
