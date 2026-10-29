#!/usr/bin/env python
"""Presentation schematics, not fitted data; watercolor artwork in Matplotlib.

Run: /home/dutra/.conda/envs/lenstronomy/bin/python scripts/make_galaxy_8971_illustrations.py
"""
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.transforms import Affine2D
from functools import lru_cache
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, INK = "tab:blue", "tab:orange", "#29333d"
# Shared illustrative luminosities; no posterior precision implied.
X = np.linspace(.08, .91, 19)
Y = .15 + .65 * X + np.array([-.02,.04,-.05,.02,.08,-.04,.05,-.07,.03,.07,-.05,.02,-.03,.06,-.04,.04,-.03,.04,-.02])


def text(ax, x, y, label, size=18, **kwargs):
    return ax.text(x, y, label, fontsize=size, color=INK, ha="center", va="center", **kwargs)


def equation(ax, x, y, label, size):
    """Use external LaTeX for equations while preserving native plot typography."""
    return text(ax, x, y, label, size, usetex=True, fontfamily="sans-serif")


def arrow(ax, start, end, color=ORANGE, curve=0):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>",
                color=color, lw=2, mutation_scale=17, connectionstyle=f"arc3,rad={curve}"))


def galaxy(ax, x, y, radius=.4, color=ORANGE, angle=30, arcs=False, style="f"):
    """Embed the selected artwork; chart elements remain vector in the PDF.

    These generated illustrations are schematic, not observations of 8971.
    RGBA assets are stored alongside this script for reproducibility.
    """
    if style == "f":
        asset = "style_f_lens.png" if arcs else "style_f_member.png"
        artwork = galaxy_artwork(asset)
    else:
        # Each transparent asset sheet contains lens/member in equal halves.
        filename = "watercolor_b_pair_subtle_lanes.png" if style == "b" else f"watercolor_{style}_pair.png"
        sheet = galaxy_artwork(filename)
        midpoint = sheet.shape[1] // 2
        artwork = sheet[:, :midpoint] if arcs else sheet[:, midpoint:]
    # Fade the illustration's outskirts at compositing time. The source PNG
    # has a nearly opaque gray rim, which transparency alone does not remove.
    # Keep the original pixels intact and preserve the blue lensing strokes.
    opacity = artwork[..., 3].copy()
    if style == "b":
        yy, xx = np.mgrid[:artwork.shape[0], :artwork.shape[1]]
        xx = (xx / (artwork.shape[1]-1) - .5)
        yy = (.5 - yy / (artwork.shape[0]-1))
        a = np.deg2rad(36)
        u, v = xx*np.cos(a)+yy*np.sin(a), -xx*np.sin(a)+yy*np.cos(a)
        r = np.hypot(u/.55, v/.29)
        t = np.clip((r-.45)/.55, 0, 1)
        feather = 1-t*t*(3-2*t)
        blue = np.clip((artwork[..., 2]-artwork[..., 0]-.05)/.25, 0, 1)
        opacity *= np.maximum(feather, blue)
    half = radius * 1.65
    transform = Affine2D().rotate_deg_around(x, y, angle - 30) + ax.transData
    ax.imshow(artwork, extent=(x-half,x+half,y-half,y+half),
              transform=transform, origin="upper", interpolation="lanczos", alpha=opacity, zorder=1)


@lru_cache(maxsize=4)
def galaxy_artwork(name):
    return plt.imread(ROOT / "scripts/assets/galaxy_8971" / name)


def canvas(compact=False):
    fig = plt.figure(figsize=(15,6.6) if compact else (16,9))
    positions = ([0,.025,.50,.955], [.48,.025,.50,.955]) if compact else ([.015,.06,.48,.88], [.505,.06,.48,.88])
    axes = [fig.add_axes(position) for position in positions]
    for ax in axes:
        ax.set(xlim=(0,8), ylim=(1.45,8.15) if compact else (0,8.25), aspect="equal")
        ax.axis("off")
    # These identify the compared models, rather than titling the plots.
    text(axes[0],4,7.9,"Lenstool model (Bergamini et al. 2023)",20,weight="bold")
    text(axes[1],4,7.9,"Our hierarchical model",20,weight="bold")
    text(axes[0],4,7.35,"Galaxy 8971 is fitted independently\nof the member scaling relation.",15,linespacing=1.35)
    text(axes[1],4,7.35,"All members follow a shared scaling relation\nwith individual deviations.",15,linespacing=1.35)
    return fig, axes


def relation(ax, origin, width, height, axes=True, galaxies=False, style="f", values=None, connect=False):
    ox, oy = origin
    values = Y if values is None else np.asarray(values)
    baseline = .15 + .65*X
    if axes:
        arrow(ax,(ox,oy),(ox+width+.15,oy),INK)
        arrow(ax,(ox,oy),(ox,oy+height+.22),INK)
        text(ax,ox+width/2,oy-.43,"Luminosity",20)
        text(ax,ox-.48,oy+height/2,"Velocity dispersion",20,rotation=90)
    if connect:
        ax.vlines(ox+width*X,oy+height*baseline,oy+height*values,
                  colors=BLUE,lw=1.0,alpha=.55,zorder=3)
    for i,(x,y) in enumerate(zip(X,values)):
        if galaxies and i%2==0:
            # Place the illustration outward from its marker, never across
            # the relation or the member's deviation connector.
            side = (1 if y >= baseline[i] else -1) if connect else (1 if i%4==0 else -1)
            displacement = side*.37
            galaxy(ax,ox+width*x,oy+height*y+displacement,.17+.065*x,
                   color="#788fa5" if i%4==0 else "#ac9981",angle=25+i*13,style=style)
    ax.plot(ox+width*np.array([.04,.97]),oy+height*(.15+.65*np.array([.04,.97])),
            color=INK,lw=1.8,zorder=4)
    ax.scatter(ox+width*X,oy+height*values,s=38,color=BLUE,edgecolors="none",zorder=5)


def member_point(ax,x,y):
    ax.errorbar(x,y,yerr=.25,fmt="D",ms=10,color=ORANGE,ecolor=ORANGE,
                elinewidth=1.7,capsize=4,zorder=6)


def minimal():
    fig,(left,right)=canvas()
    relation(left,(.85,1.55),4.25,4.85)
    member_point(left,6.55,4.65)
    text(left,6.55,5.2,"8971",20)
    for lo,hi in [(40,140),(220,300)]:
        left.add_patch(Arc((6.55,4.65),1.3,1.65,theta1=lo,theta2=hi,color=BLUE,lw=2))
    text(left,6.55,3.42,"Fit separately",17)
    relation(right,(.85,1.55),6.65,4.85)
    x=.85+6.65*.64
    baseline=1.55+4.85*(.15+.65*.64)
    y=baseline+.65
    right.plot([x,x],[baseline,y],color=ORANGE,ls=(0,(3,3)),lw=1.5)
    member_point(right,x,y)
    text(right,x+.62,y+.15,"8971",20)
    right.add_patch(Arc((x,y),1.0,1.3,theta1=45,theta2=135,color=BLUE,lw=2))
    text(right,x-.82,baseline+.53,"Allowed\ndeviation",13)
    text(left,4,.48,"Independent galaxy parameters",20)
    text(right,4,.48,"Individual deviations, shared relation",20)
    return fig


def illustrated(style="f"):
    fig,(left,right)=canvas(compact=True)
    baseline_values = .15 + .65*X
    # Fixed random schematic deviations, not posterior measurements.
    rng = np.random.default_rng(8971)
    offsets = np.clip(rng.normal(0,.12,len(X)),-.22,.22)
    deviated_values = np.clip(baseline_values+offsets,.12,.92)
    # Illustrative uncertainty in the shared relation, not individual offsets.
    band_x = np.linspace(.04,.97,160)
    band_mean = .15+.65*band_x
    band_width = np.sqrt(.035**2 + (.10*(band_x-.5))**2)
    left.fill_between(.35+6.5*band_x,1.7+3.15*(band_mean-band_width),
                      1.7+3.15*(band_mean+band_width),color=BLUE,alpha=.15,
                      linewidth=0,zorder=0)
    for ax in (left,right):
        galaxy(ax,5.9,6.05,.73,color="#c08444",arcs=True,style=style)
        relation(ax,(.35,1.7),6.5,3.15,axes=False,galaxies=True,style=style,
                 values=baseline_values if ax is left else deviated_values,connect=ax is right)
    # A crossed connection marks exclusion from the population relation.
    arrow(left,(5.55,5.02),(5.00,3.64),color="tab:red",curve=.18)
    left.plot([4.85,5.40],[4.60,4.16],color="tab:red",lw=2.2,
              solid_capstyle="round",zorder=8)
    # Same luminosities and mean relation; only our population has deviations.
    text(left,2.25,6.6,"Scaling relation",18)
    x=.35+6.5*.70
    baseline=1.7+3.15*(.15+.65*.70)
    y=baseline+.70
    right.plot([x,x],[baseline,y],color="tab:red",lw=2,ls=(0,(3,3)),zorder=6)
    right.scatter([x],[y],marker="D",s=145,color="tab:red",
                  edgecolors="tab:red",linewidths=1.8,zorder=7)
    arrow(right,(5.65,5.02),(x+.03,y+.15),color="tab:red",curve=.15)
    text(right,2.25,6.6,"Shared relation\n+ individual deviations",17,linespacing=1.2)
    # Draft equations member_scaling_mean and member_scaling_scatter,
    # combined algebraically on the right. These describe population members;
    # the independently fitted galaxy on the left is excluded.
    equation(left,2.25,5.82,r"$\displaystyle\sigma_{0,g}=\sigma_\star\left(\frac{L_g}{L_\star}\right)^{\alpha_\sigma}$",20)
    equation(left,2.25,5.06,r"$\displaystyle r_{\mathrm{cut},g}=r_{\mathrm{cut},\star}\left(\frac{L_g}{L_\star}\right)^{\beta_r}$",20)
    equation(right,2.25,5.82,r"$\displaystyle\sigma_{0,g}=\sigma_\star\left(\frac{L_g}{L_\star}\right)^{\alpha_\sigma}\,\exp(s_\sigma z_{\sigma,g})$",20)
    equation(right,2.25,5.06,r"$\displaystyle r_{\mathrm{cut},g}=r_{\mathrm{cut},\star}\left(\frac{L_g}{L_\star}\right)^{\beta_r}\,\exp(s_r z_{r,g})$",20)
    # Shared population parameters versus standardized member-level deviations.
    equation(right,5.25,2.23,
             r"\sffamily Shared hyperparameters: $\sigma_\star,\ r_{\mathrm{cut},\star},\ \alpha_\sigma,\ \beta_r,\ s_\sigma,\ s_r$",12)
    equation(right,5.25,1.92,
             r"\sffamily $s_\sigma,\ s_r$: population scatter in log space",12)
    equation(right,5.25,1.61,
             r"\sffamily $z_{\sigma,g},\ z_{r,g}$: standardized member deviations",12)
    return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"plots/presentation")
    parser.add_argument("--variant", choices=("all","minimal","illustrated"), default="all")
    parser.add_argument("--galaxy-style", choices=("f","b","c"), default="b")
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with plt.style.context(ROOT/"src/lenscluster/style.mplstyle"), plt.rc_context({
            "font.family":"DejaVu Sans","pdf.fonttype":42,"ps.fonttype":42,
            "text.latex.preamble":r"\usepackage{sfmath}",
            "savefig.bbox":None,"axes.grid":False}):
        for name,draw in [("minimal",minimal),("illustrated",illustrated)]:
            if args.variant not in ("all",name):
                continue
            fig=illustrated(args.galaxy_style) if name == "illustrated" else draw()
            suffix = f"_watercolor_{args.galaxy_style}" if name == "illustrated" and args.galaxy_style != "f" else ""
            for ext in ("pdf","png"):
                output=args.output_dir/f"galaxy_8971_{name}_python{suffix}.{ext}"
                fig.savefig(output,dpi=180,facecolor="white")
                print(output)
                if name == "illustrated" and args.galaxy_style == "b":
                    fig.savefig(args.output_dir/f"galaxy_8971_illustrated_python.{ext}",
                                dpi=180,facecolor="white")
            plt.close(fig)


if __name__=="__main__":
    main()
