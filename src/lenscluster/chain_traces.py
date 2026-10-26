"""Post-warmup chain traces and convergence diagnostics; no lens evaluations."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

STYLE = Path(__file__).with_name("style.mplstyle")
RC = {"font.family": "sans-serif", "mathtext.fontset": "dejavusans", "mathtext.default": "regular",
      "font.size": 8, "axes.labelsize": 9, "xtick.labelsize": 7, "ytick.labelsize": 7,
      "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
      "xtick.top": False, "ytick.right": False, "pdf.fonttype": 42}


def trace_data(posterior, specs):
    """Preserve recorded chain boundaries and diagnose the coordinates we plot."""
    grouped = getattr(posterior, "grouped_samples", None)
    if grouped is None or np.asarray(grouped).ndim == 0:
        return None
    grouped = np.asarray(grouped)
    samples, physical = np.asarray(posterior.samples), np.asarray(posterior.physical)
    if (grouped.ndim != 3 or grouped.shape[-1] != len(specs)
            or samples.shape != physical.shape or samples.shape != (np.prod(grouped.shape[:2]), len(specs))):
        raise ValueError("Chain traces require matching grouped, flat, and reported parameter shapes")
    if not np.array_equal(grouped.reshape(samples.shape), samples, equal_nan=True):
        raise ValueError("Saved grouped and flat samples disagree; cannot reconstruct reporting-coordinate chains")
    if not grouped.size:
        return None
    chains = physical.reshape(grouped.shape)
    n_chains, n_draws, n_parameters = chains.shape
    rhat, ess = np.full(n_parameters, np.nan), np.full(n_parameters, np.nan)
    varying = np.isfinite(chains).all(axis=(0, 1)) & (np.ptp(chains, axis=(0, 1)) > 0)
    if n_chains >= 2 and n_draws >= 4 and varying.any():
        from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
        rhat[varying] = np.asarray(split_gelman_rubin(chains[:, :, varying]))
        ess[varying] = np.asarray(effective_sample_size(chains[:, :, varying]))
    global_indices = [i for i, s in enumerate(specs)
                      if s.component_family not in {"member_scatter_offset", "source_position"}]
    # Ties retain parameter order; undefined R-hat values sort last.
    ranked = np.argsort(-np.nan_to_num(rhat, nan=-np.inf), kind="stable")
    selected = global_indices + [int(i) for i in ranked if i not in global_indices][:12]
    fields = {}
    for key in ("log_prob", "accept_prob", "num_steps", "diverging"):
        value = getattr(posterior, key, None)
        if value is not None and np.asarray(value).size == n_chains * n_draws:
            fields[key] = np.asarray(value).reshape(n_chains, n_draws)
    return {"chains": chains, "rhat": rhat, "ess": ess, "selected": selected, "fields": fields}


def _overview(pdf, data, colors, handles):
    fields = data["fields"]
    keys = [key for key in ("log_prob", "accept_prob", "num_steps") if key in fields]
    if not keys:
        return
    fig, axes = plt.subplots(len(keys), 1, figsize=(7.1, 6.5), sharex=True, squeeze=False)
    labels = {"log_prob": "Log posterior", "accept_prob": "Acceptance probability", "num_steps": "Leapfrog steps"}
    draws = np.arange(1, data["chains"].shape[1] + 1)
    for ax, key in zip(axes[:, 0], keys):
        for chain, color in enumerate(colors):
            ax.plot(draws, fields[key][chain], color=color, lw=.4, alpha=.7, rasterized=True)
        ax.set_ylabel(labels[key])
        if key == "num_steps" and np.all(fields[key] > 0):
            ax.set_yscale("log")
    if "diverging" in fields:
        divergent = fields["diverging"].astype(bool)
        counts = divergent.sum(axis=1)
        fig.text(.5, .925, "Divergences by chain: " + ", ".join(map(str, counts))
                 + f"  ({int(counts.sum())}/{divergent.size} total)", ha="center", fontsize=8,
                 color="tab:red" if counts.any() else ".2")
        for chain, color in enumerate(colors):
            locations = draws[divergent[chain]]
            axes[0, 0].plot(locations, np.full(len(locations), .025 + chain * .02), "|", color=color,
                            ms=3, alpha=.5, transform=axes[0, 0].get_xaxis_transform())
    axes[-1, 0].set_xlabel("Post-warmup draw per chain")
    fig.legend(handles=handles, loc="upper center", ncol=min(len(colors), 4), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .89))
    pdf.savefig(fig)
    plt.close(fig)


@plt.rc_context(fname=STYLE, rc=RC)
def write_chain_traces(posterior, specs, figures, tables, prefix=None, *, log=print):
    """Write the sampler overview, global traces, and twelve worst other traces."""
    data = trace_data(posterior, specs)
    if data is None:
        log("[outputs] chain traces skipped: no recorded chain samples")
        return None
    figures, tables = Path(figures), Path(tables)
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    path = figures / (f"{prefix}_chain_traces.pdf" if prefix else "chain_traces.pdf")
    chains, rhat, ess = (data[k] for k in ("chains", "rhat", "ess"))
    colors = [plt.get_cmap("tab10")(i % 10) for i in range(len(chains))]
    handles = [Line2D([], [], color=c, lw=1, label=f"Chain {i + 1}") for i, c in enumerate(colors)]
    with PdfPages(path) as pdf:
        _overview(pdf, data, colors, handles)
        for start in range(0, len(data["selected"]), 5):
            indices = data["selected"][start:start + 5]
            fig, axes = plt.subplots(len(indices), 2, figsize=(7.1, 1.35 * len(indices) + .5),
                                     squeeze=False, gridspec_kw={"width_ratios": [3.4, 1]})
            for row, j in enumerate(indices):
                ax, density = axes[row]
                values = chains[:, :, j]
                finite = values[np.isfinite(values)]
                edges = None
                if finite.size:
                    low, high = finite.min(), finite.max()
                    edges = np.linspace(low, high, 32) if high > low else np.linspace(low - .1, high + .1, 32)
                for chain, color in enumerate(colors):
                    ax.plot(np.arange(1, values.shape[1] + 1), values[chain], color=color,
                            lw=.45, alpha=.75, rasterized=True)
                    good = values[chain][np.isfinite(values[chain])]
                    if good.size:
                        density.hist(good, bins=edges, density=True, histtype="step", color=color,
                                     lw=.8, orientation="horizontal")
                ax.text(0, 1.07, specs[j].name, transform=ax.transAxes, fontsize=8, ha="left")
                info = f"Split R-hat = {rhat[j]:.3f}    ESS = {ess[j]:.0f}" if np.isfinite(rhat[j]) else "R-hat / ESS unavailable"
                ax.text(1, 1.07, info, transform=ax.transAxes, fontsize=8, ha="right",
                        color="tab:red" if rhat[j] > 1.01 else ".2")
                density.set_ylim(ax.get_ylim())
                density.tick_params(labelleft=False)
                density.set_xlabel("Density" if row == len(indices) - 1 else "")
                ax.set_xlim(.5, values.shape[1] + .5)
                ax.set_ylabel("Parameter value")
                if row < len(indices) - 1:
                    ax.tick_params(labelbottom=False)
            axes[-1, 0].set_xlabel("Post-warmup draw per chain")
            fig.legend(handles=handles, loc="upper center", ncol=min(len(colors), 4), frameon=False, fontsize=8)
            fig.tight_layout(rect=(0, 0, 1, .96), h_pad=1.2)
            pdf.savefig(fig)
            plt.close(fig)
    pd.DataFrame({"parameter": [s.name for s in specs], "split_rhat": rhat, "ess": ess}).to_csv(
        tables / "chain_convergence.csv", index=False)
    metadata = {"chains": chains.shape[0], "draws_per_chain": chains.shape[1], "warmup_included": False,
                "coordinates": "Saved reporting coordinates; noncentered member offsets remain standardized",
                "rhat": "NumPyro split Gelman-Rubin; not rank-normalized",
                "ess": "NumPyro effective_sample_size; not bulk/tail ESS",
                "selection": "All global parameters and twelve remaining parameters with largest split R-hat",
                "plotted_parameters": [specs[i].name for i in data["selected"]]}
    if "diverging" in data["fields"]:
        metadata["divergences_per_chain"] = data["fields"]["diverging"].astype(bool).sum(axis=1).tolist()
    (tables / "chain_traces.json").write_text(json.dumps(metadata, indent=2) + "\n")
    log(f"[outputs] chain traces: {path}")
    return path
