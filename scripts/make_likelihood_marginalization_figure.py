"""Illustrate latent-position integration with a normalized 1D fold pair.

This is a schematic, not the two-dimensional cluster likelihood. The source
Gaussian is conditioned on the fold's beta >= 0 domain; branches have equal
weight. No manuscript or existing candidate figures are modified.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.signal import fftconvolve
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf/likelihood_candidates"


def normal(x, mean=0., sigma=1.):
    return np.exp(-0.5*((x-mean)/sigma)**2)/(np.sqrt(2*np.pi)*sigma)


def main():
    source, scatter, measurement = .55, .55, .42
    normalization = 2*ndtr(source/scatter)

    def true_density(x):
        return abs(x)*normal(x*x/2, source, scatter)/normalization

    x = np.linspace(-7, 7, 28001)
    dx = x[1]-x[0]
    true = true_density(x)
    measured = fftconvolve(true, normal(x, sigma=measurement), mode="same")*dx
    # Place the illustrative observation at the critical curve, where the
    # pointwise true-position density vanishes but its convolution is positive.
    observation = 0.
    kernel = normal(x, observation, measurement)
    integrand = true*kernel
    value = np.trapezoid(integrand, x)
    checks = []
    for position in [-1.5, -.5, 0., .5, 1.5]:
        f = lambda u: true_density(u)*normal(position-u, sigma=measurement)
        reference = quad(f, -np.inf, 0, epsabs=1e-12)[0]+quad(f, 0, np.inf, epsabs=1e-12)[0]
        checks.append(abs(np.interp(position, x, measured)-reference))
    assert max(checks) < 1e-7
    assert abs(np.trapezoid(true, x)-1) < 1e-7
    assert abs(np.trapezoid(measured, x)-1) < 1e-7
    assert np.isclose(value, measured[len(x)//2], atol=1e-12)
    assert true[len(x)//2] == 0 and value > 0

    style = {"font.size":10, "axes.labelsize":10, "legend.fontsize":9,
             "mathtext.fontset":"dejavusans", "mathtext.default":"regular",
             "axes.linewidth":.9, "pdf.fonttype":42, "axes.grid":False}
    with plt.style.context(ROOT / "src/lenscluster/style.mplstyle"), plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.3))
        blue, orange, purple, ink = "tab:blue", "tab:orange", "tab:purple", "#252930"
        left, right = axes
        left.plot(x, true, color=ink, lw=1.6, label="True-position density")
        left.plot(x, kernel, color=blue, lw=1.6, ls="--", label="Measurement uncertainty")
        left.fill_between(x, integrand, color=orange, alpha=.32)
        left.plot(x, integrand, color=orange, lw=1.8, label="Product to integrate")
        left.set(xlabel=r"Unknown true position, $\theta$", ylabel="Density / integrand", ylim=(0, 1.32))
        left.legend(loc="upper left", handlelength=1.9)
        left.annotate("Integrate the\nshaded area", xy=(.48, .07), xytext=(1.4,.67),
                      fontsize=9, ha="center", arrowprops={"arrowstyle":"-", "color":orange, "lw":1.2})
        right.plot(x, true, color=ink, lw=1.4, ls="--", label="Before measurement error")
        right.plot(x, measured, color=orange, lw=2.2, label="Marginalized likelihood")
        right.plot(observation, value, "o", color=purple, ms=5, zorder=5)
        right.plot([0,0], [0,value], color=purple, lw=1, ls=":")
        right.annotate("Value of the\nshaded integral", xy=(0,value), xytext=(0,.50),
                       ha="center", fontsize=9, color=purple,
                       arrowprops={"arrowstyle":"-", "color":purple, "lw":1})
        right.set(xlabel=r"Measured position, $\theta^{\rm meas}$", ylabel="Probability density", ylim=(0,.78))
        right.legend(loc="upper left", handlelength=1.9)
        for ax in axes:
            ax.set_xlim(-2.5, 2.5)
            ax.set_xticks([0], ["Critical curve"])
            ax.set_yticks([])
            ax.minorticks_off()
            ax.tick_params(axis="x", labelsize=9, length=3, width=.9, top=False)
        fig.subplots_adjust(left=.07, right=.985, bottom=.31, top=.97, wspace=.27)
        fig.text(.52, .055,
                 r"$p(\theta^{\rm meas})=\int p(\theta^{\rm meas}\mid\theta)\,p(\theta)\,d\theta$",
                 ha="center", fontsize=12)
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / "likelihood_marginalization.pdf"
        fig.savefig(path)
        plt.close(fig)
    report = {"example":"normalized equal-branch 1D fold, beta=theta^2/2",
              "max_absolute_quadrature_difference":max(checks),
              "integrated_true_density":float(np.trapezoid(true,x)),
              "integrated_measured_density":float(np.trapezoid(measured,x)),
              "likelihood_at_critical_curve":float(value)}
    path.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n")
    print(path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
