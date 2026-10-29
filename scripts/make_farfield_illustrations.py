"""Synthetic far-field illustrations, using the circular dPIE series in farfield.py.

Run with the ggsl, hdm or lenstronomy conda Python interpreter.
Outputs vector PDFs and PNG previews. No cluster-fit results are used.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, Rectangle, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "output/pdf/farfield_candidates"
BLUE, GREEN, RED, PURPLE, INK = "#1674A4", "#218778", "#C44E35", "#77529B", "#252930"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "mathtext.fontset": "stix", "axes.edgecolor": INK,
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": False, "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / (name + ".pdf"), bbox_inches="tight", pad_inches=.15)
    fig.savefig(OUT / (name + ".png"), bbox_inches="tight", pad_inches=.15, dpi=180)
    plt.close(fig)


def arrow(ax, start, end, color=INK, width=1.4, style="-|>", scale=13, **kw):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=scale,
                                linewidth=width, color=color, **kw))


def geometry():
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.set(xlim=(-1.2, 10), ylim=(-3.0, 2.7), aspect="equal")
    ax.axis("off")
    galaxy = np.array([-.1, .1])
    images = np.array([[6.4, .9], [7.3, 1.5], [7.7, -.3], [6.9, -1], [5.8, -.2]])
    ax.add_patch(Ellipse((6.8,.25), 3.2, 3.4, angle=-22, facecolor=BLUE,
                         alpha=.035, edgecolor="none"))
    for rad, alpha in [(.52,.04), (.36,.08), (.20,.16)]:
        ax.add_patch(Circle(galaxy, rad, facecolor=PURPLE, alpha=alpha, edgecolor="none"))
    ax.add_patch(Circle(galaxy, .36, fill=False, edgecolor=PURPLE, lw=1.5))
    ax.plot(*galaxy, "o", color=PURPLE, ms=6)
    for im in images:
        ax.plot([galaxy[0],im[0]], [galaxy[1],im[1]], color=GREEN, lw=.8, alpha=.22)
        ax.plot(*im, "o", mfc="white", mec=BLUE, mew=1.5, ms=7)
    target=images[0]
    arrow(ax, galaxy, target, GREEN, width=1.7, shrinkA=8, shrinkB=7)
    mid=(galaxy+target)/2
    ax.text(mid[0],mid[1]+.22,r"$R_{ig}$",color=GREEN,fontsize=18,ha="center")
    direction=(target-galaxy)/np.linalg.norm(target-galaxy)
    arrow(ax,target+.13*direction,target+1.0*direction,RED)
    ax.text(7.55,1.22,r"$\hat{\mathbf{r}}_{ig}$".replace(r"\mathbf",r"\mathbf"),
            color=RED,fontsize=17)
    arrow(ax,galaxy,galaxy+np.array([0,.36]),PURPLE,width=1)
    ax.text(-.26,.62,r"$r_{\rm cut}$",color=PURPLE,ha="center",fontsize=14)
    ax.text(-.1,-.7,r"Member $g$",ha="center",color=PURPLE)
    ax.text(6.8,2.35,r"Observed images $\mathbf{\theta}_i$".replace(r"\mathbf",r"\mathbf"),
            ha="center",color=BLUE)
    ax.text(1.25,-1.7,r"$\mathbf{\alpha}_g(\mathbf{\theta}_i)\simeq"
            r"\sum_{n\in\{1,2,4\}}$",fontsize=20,ha="center",va="center")
    ax.text(5.2,-1.7,r"$w_{g,n}(\sigma_{0,g},\,r_{{\rm core},g},\,r_{{\rm cut},g})$",
            color=PURPLE,fontsize=20,ha="center",va="center")
    ax.text(8.5,-1.42,r"$\hat{\mathbf{r}}_{ig}$",
            color=RED,fontsize=20,ha="center",va="center")
    ax.plot([8.18,8.82],[-1.73,-1.73],color=INK,lw=1)
    ax.text(8.5,-2.0,r"$R_{ig}^{\,n}$",
            color=GREEN,fontsize=20,ha="center",va="center")
    for left,right,color in [(3.15,7.25,PURPLE),(7.85,9.15,GREEN)]:
        ax.plot([left,left,right,right],[-2.12,-2.23,-2.23,-2.12],
                color=color,lw=1.1)
    ax.text(5.2,-2.53,"From sampled member parameters",color=PURPLE,
            fontsize=11,ha="center")
    ax.text(8.5,-2.53,"Precomputed",color=GREEN,fontsize=11,ha="center")
    save(fig,"farfield_geometry")


def matrix():
    fig,ax=plt.subplots(figsize=(12,5.2))
    ax.set(xlim=(-.6,12.4),ylim=(-1.9,4.5))
    ax.axis("off")
    # Small schematic matrix: each member contributes three geometry columns.
    rng=np.random.default_rng(8)
    values=rng.uniform(.12,.9,(6,9))
    x,y,dx,dy=.2,.6,.31,.38
    for row in range(6):
        for col in range(9):
            ax.add_patch(Rectangle((x+col*dx,y+row*dy),dx-.025,dy-.025,
                                  facecolor=GREEN,alpha=values[row,col]*.6,linewidth=0))
    for col in (3,6):
        ax.plot([x+col*dx-.013]*2,[y-.04,y+6*dy+.04],color="white",lw=3)
    ax.text(1.6,3.25,r"$\mathbf{G}$",color=GREEN,fontsize=24,ha="center")
    ax.text(1.6,3.85,"Precompute once",color=GREEN,ha="center")
    for j in range(3):
        ax.text(x+(j*3+1.5)*dx,.3,rf"$g={j+1}$",ha="center",fontsize=11)
    ax.text(1.6,-.25,r"$\hat{\mathbf{r}}_{ig}/R_{ig},\quad"
            r"\hat{\mathbf{r}}_{ig}/R_{ig}^{2},\quad\hat{\mathbf{r}}_{ig}/R_{ig}^{4}$",
            fontsize=16,ha="center")
    ax.text(3.5,1.7,r"$\times$",fontsize=26,ha="center")
    for j in range(9):
        ax.add_patch(Rectangle((4.05,.6+j*2.28/9),.5,.22,
                              facecolor=PURPLE,alpha=.25+.06*(j%3),lw=0))
    ax.text(4.3,3.25,r"$\mathbf{w}$",color=PURPLE,fontsize=24,ha="center")
    ax.text(4.3,3.85,"Update each evaluation",color=PURPLE,ha="center")
    ax.text(4.3,-.8,r"$\sigma_g,\ r_{{\rm core},g},\ r_{{\rm cut},g}$",
            ha="center",fontsize=15,color=PURPLE)
    ax.text(5.1,1.7,r"$=$",fontsize=26)
    for j in range(6):
        ax.add_patch(Rectangle((5.85,.6+j*dy),.5,dy-.025,
                              facecolor=BLUE,alpha=.25+.09*j,lw=0))
    ax.text(6.1,3.25,r"$\mathbf{\alpha}_{\rm far}$",fontsize=24,color=BLUE,ha="center")
    ax.text(6.1,-.25,"Sum over members",ha="center",color=BLUE)
    ax.plot([7.25,7.25],[-.55,3.9],color=INK,alpha=.12,lw=1)
    ax.text(9.65,3.3,"The same weights give",ha="center")
    ax.text(9.65,2.35,r"$\nabla\mathbf{\alpha}_{\rm far}\simeq\mathbf{G}^{(1)}\mathbf{w}$",
            fontsize=21,ha="center")
    ax.text(9.65,1.15,r"$\nabla^2\mathbf{\alpha}_{\rm far}\simeq\mathbf{G}^{(2)}\mathbf{w}$",
            fontsize=21,ha="center")
    ax.text(9.65,.1,"Geometry and its derivatives stay fixed.",ha="center",fontsize=10)
    ax.text(5.8,-1.6,"Member properties vary throughout inference; their geometric factors are reused.",
            ha="center",fontsize=11)
    save(fig,"farfield_matrix")


def accuracy():
    # Radius in units of cut radius; alpha in units M/r_cut.
    core,cut=.05,1.
    radius=np.geomspace(1,30,500)
    exact=(1-(core+cut)/(np.hypot(radius,core)+np.hypot(radius,cut)))/radius
    first=1/radius
    second=first-(core+cut)/(2*radius**2)
    fourth=second+(core+cut)*(core*core+cut*cut)/(8*radius**4)
    errors=[abs(a-exact)/exact for a in (first,second,fourth)]
    if not all(e[-1] < e[0] for e in errors):
        raise ValueError("Far-field errors do not decrease")
    if not errors[2][-1] < errors[1][-1] < errors[0][-1]:
        raise ValueError("Unexpected truncation ordering")
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    axes[0].loglog(radius,exact,color=INK,lw=2,label="Exact circular dPIE")
    axes[0].loglog(radius,fourth,color=GREEN,lw=1.7,ls="--",label=r"$R^{-1}+R^{-2}+R^{-4}$")
    axes[0].set(xlabel=r"$R/r_{\rm cut}$",ylabel=r"$\alpha/(M/r_{\rm cut})$")
    axes[0].legend(frameon=False,loc="lower left",fontsize=10)
    for err,color,label in zip(errors,[BLUE,RED,GREEN],
                               [r"$R^{-1}$",r"$R^{-1}+R^{-2}$",r"$R^{-1}+R^{-2}+R^{-4}$"]):
        axes[1].loglog(radius,err,color=color,lw=1.8,label=label)
    axes[1].set(xlabel=r"$R/r_{\rm cut}$",ylabel=r"$|\alpha_{\rm approx}-\alpha_{\rm exact}|/\alpha_{\rm exact}$")
    axes[1].legend(frameon=False,fontsize=10)
    for ax in axes:
        ax.set_xlim(1,30)
        ax.tick_params(which="both",direction="out")
    fig.subplots_adjust(wspace=.32,bottom=.22)
    fig.text(.5,.015,r"Circular dPIE example, $r_{\rm core}/r_{\rm cut}=0.05$; coefficients are included in each series.",
             ha="center",fontsize=10)
    save(fig,"farfield_convergence")
    return dict(core_to_cut=core/cut,radius_over_cut=30,
                relative_errors_at_30=[float(e[-1]) for e in errors])


def main():
    geometry()
    matrix()
    checks=accuracy()
    (OUT/"parameters.json").write_text(json.dumps(checks,indent=2)+"\n")
    (OUT/"README.txt").write_text(
        "All figures are synthetic illustrations, not fitted cluster results.\n"
        "Geometry: distant circular member and fixed image positions; separation alone is not a classification rule.\n"
        "Matrix: schematic blocks (not numerical lens matrices), grouped into three columns per member.\n"
        "Convergence: exact normalized circular dPIE compared with the implemented powers 1, 2, 4.\n"
        "The convergence example does not include elliptical-to-circular errors or the combined-member error budget.\n")
    print("Created geometry, matrix and convergence figures in",OUT)
    print("Circular-series checks:",checks)


if __name__=="__main__":
    main()
