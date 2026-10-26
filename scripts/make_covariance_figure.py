"""Covariance decomposition and its minimum-width floor near a critical curve.

Run with /home/dutra/.conda/envs/lenstronomy/bin/python.
Writes only plots/paper/covariance.pdf, independent of the working directory.
The parameters are illustrative, not a fitted cluster. Every uncertainty contour
is a unit-Mahalanobis contour (39.3% enclosed probability in two dimensions).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from matplotlib.transforms import Bbox
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "plots" / "paper" / "covariance.pdf"
STYLE = ROOT / "src" / "lenscluster" / "style.mplstyle"
BLUE, ORANGE, INK = "#1674A4", "#C44E35", "#252930"
PURPLE, MUTED = "#77529B", "#626971"
ANGLE = 25.0
# Drawing parameters only: chosen for legibility, never displayed as inferred values.
SIGMA_OBS, SIGMA_SRC = 0.1, 0.03  # arcsec, per coordinate
LAMBDA_STRETCH, LAMBDA_PERP = 0.12, 0.8
E = np.array([np.cos(np.deg2rad(ANGLE)), np.sin(np.deg2rad(ANGLE))])
N = np.array([-E[1], E[0]])
A = LAMBDA_STRETCH * np.outer(E, E) + LAMBDA_PERP * np.outer(N, N)
C_OBS = SIGMA_OBS**2 * A @ A.T
C_SRC = SIGMA_SRC**2 * np.eye(2)
C_TOTAL = C_OBS + C_SRC
SCALE = 9.6  # Source-plane display scale; all overlaid contours use the same scale.


def verify_geometry():
    """Independent eigenvalue and mapped-contour checks of the plotted geometry."""
    np.testing.assert_allclose(A, A.T, rtol=0, atol=1e-16)
    np.testing.assert_allclose(A @ E, LAMBDA_STRETCH * E, atol=1e-16)
    np.testing.assert_allclose(A @ N, LAMBDA_PERP * N, atol=1e-16)
    values, vectors = np.linalg.eigh(C_TOTAL)
    expected = SIGMA_SRC**2 + SIGMA_OBS**2 * np.array([LAMBDA_STRETCH, LAMBDA_PERP])**2
    np.testing.assert_allclose(values, expected, rtol=2e-15)
    np.testing.assert_allclose(np.sqrt(np.linalg.eigvalsh(C_OBS)), SIGMA_OBS*np.abs([LAMBDA_STRETCH, LAMBDA_PERP]), rtol=2e-15)
    np.testing.assert_allclose(vectors[:, 1] @ E, 0.0, atol=2e-15)
    phase = np.linspace(0, 2*np.pi, 361)
    mapped = A @ (SIGMA_OBS * np.stack((np.cos(phase), np.sin(phase))))
    np.testing.assert_allclose(np.einsum('in,ij,jn->n', mapped, np.linalg.inv(C_OBS), mapped),
                               1.0, rtol=2e-14, atol=2e-14)
    # An isotropic source floor prevents the narrow axis from vanishing at a critical curve.
    critical_a = LAMBDA_PERP * np.outer(N, N)
    critical_c = SIGMA_SRC**2*np.eye(2) + SIGMA_OBS**2*critical_a@critical_a.T
    np.testing.assert_allclose(np.linalg.eigvalsh(critical_c)[0], SIGMA_SRC**2, rtol=2e-15)
    return np.sqrt(values)


def arrow(ax, start, end, *, color=INK, lw=0.9, style="-|>", size=8, zorder=6):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=size,
                                linewidth=lw, color=color, shrinkA=0, shrinkB=0, zorder=zorder))


def contour(ax, center, covariance, *, color, lw=1.3, linestyle="solid", zorder=5, scale=SCALE):
    values, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, 1]
    angle = np.rad2deg(np.arctan2(direction[1], direction[0]))
    ax.add_patch(Ellipse(center, 2*scale*np.sqrt(values[1]), 2*scale*np.sqrt(values[0]),
                         angle=angle, facecolor="none", edgecolor=color, linewidth=lw,
                         linestyle=linestyle, zorder=zorder))


def arc(ax, center):
    """Contextual arc, locally tangent to E; not a ray-traced cluster image."""
    t = np.linspace(-1.0, 1.0, 240)
    curve = center + t[:, None]*E + .10*t[:, None]**2*N
    taper = np.maximum(0, 1-t**2)**.7
    for width, shade in ((.115, "#E8E8E8"), (.082, "#D0D0D0"), (.046, "#B5B5B5")):
        edge = width*taper[:, None]*N
        ax.add_patch(Polygon(np.concatenate((curve+edge, (curve-edge)[::-1])),
                             closed=True, facecolor=shade, edgecolor="none", zorder=1))


def label(ax, xy, text, **kwargs):
    defaults = dict(ha="center", va="center", fontsize=plt.rcParams["font.size"], color=INK, zorder=12)
    defaults.update(kwargs)
    return ax.text(*xy, text, **defaults)


def make_figure():
    semiaxes = verify_geometry()
    # Verify the complete plotted curve from independently diagonalized covariances.
    t = np.linspace(0, 3, 601)
    s_min = t * SIGMA_SRC / SIGMA_OBS
    s_max = max(LAMBDA_PERP, 1.1 * s_min.max())
    jacobians = s_min[:, None, None]*np.outer(E, E) + s_max*np.outer(N, N)
    covariances = SIGMA_SRC**2*np.eye(2) + SIGMA_OBS**2*(jacobians @ jacobians.swapaxes(-1, -2))
    widths = np.sqrt(np.linalg.eigvalsh(covariances)[:, 0])/SIGMA_SRC
    expected = np.hypot(1.0, t)
    curve_error = np.max(np.abs(widths - expected))
    np.testing.assert_allclose(widths, expected, rtol=3e-15, atol=3e-15)
    example_t = SIGMA_OBS * LAMBDA_STRETCH / SIGMA_SRC
    np.testing.assert_allclose(semiaxes[0]/SIGMA_SRC, np.hypot(1.0, example_t), rtol=3e-15)
    plt.style.use(str(STYLE))
    # Scale the shared typography for this compact multi-diagram figure.
    plt.rcParams.update({
        "font.size": 9.5,
        "legend.fontsize": 9,
        "text.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.grid": False, "savefig.facecolor": "white",
    })
    annotation_size = plt.rcParams["legend.fontsize"]
    fig = plt.figure(figsize=(6.3, 2.3), facecolor="white")
    ax = fig.add_axes([0, .10, 4.5085/6.3, .80],
                      xlim=(0, 6.5), ylim=(-.2, 2.45), aspect="equal")
    ax.set_axis_off()
    image = np.array([1.08, 1.40])
    total = np.array([3.40, 1.40])


    arc(ax, image)
    contour(ax, image, SIGMA_OBS**2*np.eye(2), color=BLUE, lw=1.5, scale=6.0)
    arrow(ax, image-.85*E, image+.85*E, color=PURPLE, style="<->", lw=1.1)
    arrow(ax, image, image+.73*N, color=MUTED, lw=.8)
    label(ax, image+.90*E-.17*N, r'$\boldsymbol{e}_{\rm arc}$', color=PURPLE,
          fontsize=annotation_size, ha="center", rotation=ANGLE)
    ax.plot(*image, "o", ms=3.2, color=INK, zorder=10)
    arrow(ax, image, image+np.array([0, -.60]), color=BLUE, lw=.9)
    label(ax, image+np.array([.25, -.28]), r'$\sigma_{\rm obs}$', color=BLUE, fontsize=annotation_size)

    # Local mapping: this is a differential map, not an inverse-Jacobian image-error ellipse.
    arrow(ax, (2.05, 1.40), (2.75, 1.40), color=MUTED, lw=1.7, size=9)
    label(ax, (2.40, 1.22),
          r'$\delta\boldsymbol{\beta}=\boldsymbol{A}_i\,\delta\boldsymbol{\theta}$',
          fontsize=8)

    # Component outlines share the total's center and scale; covariance matrices add, not radii.
    contour(ax, total, C_OBS, color=BLUE, lw=1.8, zorder=5)
    contour(ax, total, C_SRC, color=ORANGE, lw=1.8, linestyle=(0, (3, 1.8)), zorder=6)
    contour(ax, total, C_TOTAL, color=INK, lw=1.3, zorder=4)
    # Match arrow tips to the total covariance contour, not arbitrary direction lengths.
    minor, major = SCALE * semiaxes
    for direction, radius in ((E, minor), (N, major)):
        endpoint = radius * direction / SCALE
        np.testing.assert_allclose(endpoint @ np.linalg.solve(C_TOTAL, endpoint), 1.0, rtol=2e-15)
    arrow(ax, total, total+minor*E, color=PURPLE, lw=.95)
    ax.plot(*total, "o", ms=2.8, color=INK, zorder=10)
    label(ax, total+.30*E+np.array([.045, .08]),
          r'$\boldsymbol{e}_{\rm arc}$', color=PURPLE,
          fontsize=annotation_size, ha="left")
    label(ax, total+.48*E+np.array([-.025, -.12]),
          r'$w_{\rm arc}$', color=PURPLE,
          fontsize=annotation_size, ha="left")
    label(ax, (1.08, .42), r'$\sigma_{\rm obs}^{2}\boldsymbol{I}$',
          color=BLUE, fontsize=annotation_size)
    terms = [TextArea(symbol, textprops={'color': color, 'fontsize': 8})
             for symbol, color in (
                 (r'$\boldsymbol{C}_i =$', INK),
                 (r'$\sigma_{\rm obs}^{2}\boldsymbol{A}_i\boldsymbol{A}_i^{\mathsf{T}}$', BLUE),
                 (r'$+$', INK),
                 (r'$\sigma_{\rm src}^{2}\boldsymbol{I}$', ORANGE),
             )]
    equation = HPacker(children=terms, align='baseline', pad=0, sep=3)
    ax.add_artist(AnnotationBbox(equation, (3.40, .42), frameon=False,
                                box_alignment=(.5, .5), pad=0))

    # Axis labels carry the normalization; definitions belong in the caption.
    metric = fig.add_axes([3.50/6.3, .31, 1.775/6.3, .52])
    metric.spines[['top', 'right']].set_visible(False)
    metric.spines[['bottom', 'left']].set_color(MUTED)
    metric.spines[['bottom', 'left']].set_linewidth(.8)
    metric.tick_params(axis='both', labelsize=8, colors=MUTED, direction='out',
                       top=False, right=False, length=3, width=.8)
    metric.minorticks_off()
    metric.set(xlim=(0, 3), ylim=(0, 3.5), xticks=[0, 1, 2, 3], yticks=[0, 1, 2, 3])
    metric.plot(t, t, color=BLUE, lw=1.1, linestyle=(0, (1.5, 2)))
    metric.axhline(1, color=ORANGE, lw=1.2, linestyle=(0, (3, 1.8)))
    metric.plot(t, expected, color=INK, lw=1.8)
    metric.plot(example_t, np.hypot(1, example_t), marker='o', linestyle='none',
                markerfacecolor='white', markeredgecolor=PURPLE,
                markersize=5, markeredgewidth=1.0, zorder=6)
    metric.text(2.95, .68, 'Source-scatter floor', color=ORANGE, fontsize=7, ha='right')
    line_angle = metric.transData.transform_angles(np.array([45.]), np.array([[2.1, 2.1]]))[0]
    metric.text(2.28, 1.96, 'Astrometry alone', color=BLUE, fontsize=7, ha='center',
                rotation=line_angle, rotation_mode='anchor')
    metric.text(2.12, 2.58, 'Total uncertainty', color=INK, fontsize=7,
                ha='center', rotation=line_angle, rotation_mode='anchor')
    metric.set_xlabel(r'$\sigma_{\rm obs}\,s_{\rm arc}/\sigma_{\rm src}$', fontsize=annotation_size, labelpad=3)
    metric.set_ylabel(r'$w_{\rm arc}/\sigma_{\rm src}$', fontsize=annotation_size, labelpad=4)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Trim the outer margins on all sides, preserving the relative glyph scales.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Use visible artwork, not the hidden schematic axes' empty rectangular extent.
    artists = [*ax.texts, *ax.patches, *ax.lines, *ax.artists]
    content = Bbox.union([artist.get_window_extent(renderer) for artist in artists]
                         + [metric.get_tightbbox(renderer)])
    content = content.transformed(fig.dpi_scale_trans.inverted())
    crop = Bbox.from_extents(content.x0 - .015, content.y0 - .015,
                             content.x1 + .015, content.y1 + .015)
    fig.savefig(OUT, bbox_inches=crop, metadata={"Title": "Anisotropic positional covariance and its minimum-width floor",
                              "Subject": "Illustrative unit-Mahalanobis contours; 39.3 percent enclosed probability in 2D",
                              "Creator": "scripts/make_covariance_figure.py"})
    plt.close(fig)
    print(f"Verified source semiaxes: {semiaxes[0]:.12f}, {semiaxes[1]:.12f} arcsec")
    print(f"Verified covariance axis ratio: {semiaxes[1]/semiaxes[0]:.12f}")
    print(f"Verified minimum-width relation at {len(t)} points; max absolute error={curve_error:.3e}")
    print(OUT)


if __name__ == "__main__":
    make_figure()
