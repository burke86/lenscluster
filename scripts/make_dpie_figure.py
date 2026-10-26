"""Illustrate the paper's circularized dPIE geometry and convergence.

Run with /home/dutra/.conda/envs/lenstronomy/bin/python.
Writes only plots/paper/dpie_profile.pdf, from any working directory.
The shape and scale ratio are illustrative, not fitted measurements.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "plots" / "paper" / "dpie_profile.pdf"
BLUE = "#28799B"
ORANGE = "#C26935"
INK = "#24272B"
MUTED = "#626971"
Q = 0.65
PHI_DEG = 28.0
A = 0.1  # a/s; the plotted cut radius is s=1.
S = 1.0


def convergence(radius, a=A, s=S):
    """kappa/kappa0, factored to avoid subtraction at large radius."""
    radius = np.asarray(radius)
    wa = np.hypot(a, radius)
    ws = np.hypot(s, radius)
    return a * s * (a + s) / (wa * ws * (wa + ws))


def log_slope(radius, a=A, s=S):
    """Exact d ln(kappa)/d ln(radius) of the factored profile."""
    radius = np.asarray(radius)
    wa, ws = np.hypot(a, radius), np.hypot(s, radius)
    return -radius**2 * (1 / wa**2 + 1 / ws**2 + 1 / (wa * ws))


def verify_geometry_and_profile():
    theta = np.linspace(0, 2 * np.pi, 721)
    for radius in (A, S):
        x = radius / np.sqrt(Q) * np.cos(theta)
        y = radius * np.sqrt(Q) * np.sin(theta)
        np.testing.assert_allclose(Q * x*x + y*y / Q, radius**2,
                                   rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(np.pi * radius/np.sqrt(Q) * radius*np.sqrt(Q),
                                   np.pi * radius**2)
    np.testing.assert_allclose(convergence(0), 1)
    radius = np.geomspace(1e-4, 100, 1001)
    assert np.all(np.diff(convergence(radius)) < 0)
    np.testing.assert_allclose(log_slope(1e-7), 0, atol=2e-12)
    np.testing.assert_allclose(log_slope(1e5), -3, atol=2e-9)
    # An extended -1 interval requires stronger core/cut separation.
    np.testing.assert_allclose(log_slope(1e-3, a=1e-6, s=1), -1, atol=.002)
    from lenstronomy.LensModel.Profiles.dpie_nie_e import DPIE_NIE_E
    e = (1-Q)/(1+Q)
    # On the major axis, X=xi/sqrt(q).
    h = DPIE_NIE_E.hessian(radius/np.sqrt(Q), np.zeros_like(radius), 1, A, S, e, 0)
    np.testing.assert_allclose((h[0]+h[3])/2, convergence(radius),
                               rtol=3e-11, atol=2e-14)


def arrow(ax, start, end, *, color=INK, style="-|>", lw=.8, scale=9):
    patch = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=scale,
                            linewidth=lw, color=color, shrinkA=0, shrinkB=0,
                            zorder=6)
    ax.add_patch(patch)


def make_figure():
    verify_geometry_and_profile()
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 10.5, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .7,
    })
    fig = plt.figure(figsize=(7.25, 3.8))
    geo = fig.add_axes([.025, .14, .46, .76])
    radial = fig.add_axes([.60, .20, .37, .66])
    geo.set(xlim=(-1.55, 1.58), ylim=(-1.28, 1.30), aspect="equal")
    geo.set_axis_off()
    angle = np.deg2rad(PHI_DEG)
    major = np.array([np.cos(angle), np.sin(angle)])
    minor = np.array([-np.sin(angle), np.cos(angle)])

    # Exact contours, with no potential ellipticity or perspective distortion.
    geo.add_patch(Circle((0, 0), S, fill=False, ec="#B4B9BE", lw=.8,
                         ls=(0, (2, 2.5)), zorder=1))
    for radius, color, lw in [(S, ORANGE, 1.7), (A, BLUE, 1.45)]:
        geo.add_patch(Ellipse((0,0), 2*radius/np.sqrt(Q), 2*radius*np.sqrt(Q),
                              angle=PHI_DEG, fill=False, ec=color, lw=lw, zorder=4))
    geo.plot(*np.array([-1.38*major, 1.45*major]).T, color="#B4B9BE",
             lw=.65, ls=(0,(4,3)), zorder=1)
    geo.plot(*np.array([-1.03*minor, 1.10*minor]).T, color="#B4B9BE",
             lw=.65, ls=(0,(4,3)), zorder=1)
    arrow(geo, 1.32*major, 1.46*major, color=MUTED)
    arrow(geo, .97*minor, 1.10*minor, color=MUTED)
    geo.text(*(1.53*major), "$X$", ha="center", va="center", color=MUTED)
    geo.text(*(1.18*minor), "$Y$", ha="center", va="center", color=MUTED)
    arrow(geo, (0,0), (1.42,0), color=MUTED, lw=.6)
    arrow(geo, (0,0), (0,1.13), color=MUTED, lw=.6)
    geo.text(1.48, -.025, "$x$", va="center", color=MUTED)
    geo.text(.045, 1.16, "$y$", va="bottom", color=MUTED)
    geo.add_patch(Arc((0,0), .88,.88, theta1=0, theta2=PHI_DEG, color=INK, lw=.8))
    geo.text(.51*np.cos(angle/2), .51*np.sin(angle/2), r"$\phi$", fontsize=12,
             va="center", ha="center")
    # Put dimensions on negative principal axes to leave phi unobscured.
    arrow(geo, (0,0), -S/np.sqrt(Q)*major, color=ORANGE, style="<->")
    arrow(geo, (0,0), -S*np.sqrt(Q)*minor, color=ORANGE, style="<->")
    geo.text(*(-.72*major+.12*minor), r"$s/\sqrt{q}$", color=ORANGE,
             rotation=PHI_DEG, ha="center", va="bottom", fontsize=11)
    geo.text(*(-.48*minor+.14*major), r"$s\sqrt{q}$", color=ORANGE,
             rotation=PHI_DEG-90, ha="center", va="center", fontsize=11)
    geo.plot(0,0,"o",ms=3,color=INK,zorder=8)
    geo.annotate(r"$\xi=a$", xy=A/np.sqrt(Q)*major, xytext=(.65,1.00),
                 color=BLUE, fontsize=11, ha="center",
                 arrowprops=dict(arrowstyle="-", color=BLUE, lw=.75,
                                 connectionstyle="angle3,angleA=0,angleB=70"))
    geo.text(-1.31, -.87, r"$\xi=s$", color=ORANGE, fontsize=11)
    geo.annotate(r"Equal area: $\pi s^2$", xy=(-.97,.24), xytext=(-1.48,.92),
                 color=MUTED, fontsize=9, ha="left",
                 arrowprops=dict(arrowstyle="-",color="#A5ABB1",lw=.7))
    geo.text(0,-1.20,r"$\xi^2=qX^2+Y^2/q$",ha="center",fontsize=12)

    radius = np.geomspace(1e-3, 30, 1200)
    radial.loglog(radius, convergence(radius), color=INK, lw=1.8, zorder=4)
    for value, color, label in [(A, BLUE, r"$a$"), (S, ORANGE, r"$s$")]:
        radial.axvline(value, color=color, lw=.85, ls=(0,(3,3)), zorder=1)
        radial.plot(value, convergence(value), "o", color=color, ms=4, zorder=5)
        radial.text(value, 1.6, label, color=color, ha="center", va="bottom", fontsize=12)
    radial.text(.002,1.35,r"$\kappa(0)=\kappa_0$",color=INK,fontsize=10)
    # Detached asymptotic guides, not fitted slopes over finite intervals.
    radial.plot([.0016,.018],[.62,.62],color=MUTED,lw=.85)
    radial.text(.0047,.34,r"$0$",ha="center",color=MUTED,fontsize=10)
    mid=np.array([.17,.65])
    radial.plot(mid,.037/mid,color=MUTED,lw=.85)
    radial.text(.32,.035,r"$-1$",ha="center",color=MUTED,fontsize=10)
    outer=np.array([2.,9.])
    radial.plot(outer,.018/outer**3,color=MUTED,lw=.85)
    radial.text(2.5,1e-4,r"$-3$",color=MUTED,fontsize=10)
    radial.set(xlim=(1e-3,30),ylim=(1e-6,2.6),
               xlabel=r"Circularized radius $\xi/s$",ylabel=r"$\kappa(\xi)/\kappa_0$")
    radial.set_xticks([1e-3,1e-2,.1,1,10])
    radial.set_yticks([1,1e-2,1e-4,1e-6])
    radial.tick_params(which="both",direction="in",labelsize=9)
    radial.tick_params(which="minor",length=2)
    radial.text(.50,-.25,"Grey guides: asymptotic slopes",transform=radial.transAxes,
                ha="center",fontsize=8.5,color=MUTED)
    fig.text(.035,.93,"(a)",fontsize=11)
    fig.text(.555,.93,"(b)",fontsize=11)
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUTPUT)
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    make_figure()
