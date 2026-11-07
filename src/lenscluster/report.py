"""Plain-text reports of the member classification and the posterior audit."""
from __future__ import annotations

from typing import Any

import numpy as np

from .model import ModelState


def _fmt(value: Any, width: int = 9, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return f"{'inf' if value is not None else '-':>{width}}"
    return f"{float(value):{width}.{digits}g}"


def _rank_table(rankings: list[dict[str, Any]], title: str) -> list[str]:
    lines = [title, f"  {'member':<14}{'class':<10}{'score':>9}{'truncation':>11}{'ellipticity':>12}{'r_cut\"':>9}{'min R\"':>9}{'R/r_cut':>9}"]
    for r in rankings:
        ratio = r["min_separation_arcsec"] / r["cut_radius_arcsec"] if r["cut_radius_arcsec"] > 0 else float("inf")
        lines.append(f"  {r['member']:<14}{'exact' if r['exact'] else 'far-field':<10}{_fmt(r['score'])}{_fmt(r['truncation'], 11)}"
                     f"{_fmt(r['ellipticity'], 12)}{_fmt(r['cut_radius_arcsec'])}{_fmt(r['min_separation_arcsec'])}{_fmt(ratio)}")
    return lines


def _duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "-"
    seconds = int(round(seconds))
    hours, minutes, secs = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    return f"{hours}h {minutes:02d}m {secs:02d}s" if hours else (f"{minutes}m {secs:02d}s" if minutes else f"{secs}s")


def run_summary_table(result, config) -> str:
    """The end-of-run summary: data and model sizes, image recovery and residuals at the best fit, sampler health,
    the far-field audit, and how long each stage took.  Rendered with rich without colour, so the log and the terminal
    read the same."""
    from rich.console import Console
    from rich.table import Table
    state, posterior, audit = result.state, result.posterior, result.audit
    fit = (result.summary or {}).get("fit", {})
    diagnostics = posterior.diagnostics or {}
    adaptation = diagnostics.get("adaptation", {})
    timing = result.timing or {}

    def number(value, digits=3):
        return "-" if value is None or (isinstance(value, float) and not np.isfinite(value)) else f"{value:.{digits}f}"

    sections = [
        ("Run", [("name", str(config.paths.run_name)), ("output", str(result.paths.final.root)),
                 ("stages", ("init resumed, " if result.init.resumed else "init run, ") + ("final resumed" if result.resumed else "final run"))]),
        ("Data and model", [("images, families", f"{state.images.n_images}, {state.images.n_families}"),
                            ("members: exact, far-field", f"{len(state.members.exact_positions)}, {len(state.members.far_positions)}"),
                            ("sampled parameters", str(state.n_parameters)),
                            ("global parameters, dof", f"{fit.get('n_global_parameters', '-')}, {fit.get('dof', '-')}")]),
        ("Images at the best fit", [("recovered / observed", f"{fit.get('recovered_image_count', '-')} / {fit.get('observed_image_count', '-')}"),
                                    ("missing, extra", f"{fit.get('missing_image_count', '-')}, {fit.get('extra_image_count', '-')}"),
                                    ("families fully recovered", f"{fit.get('families_fully_recovered', '-')} / {state.images.n_families}"),
                                    ("residual rms, median, max (arcsec)", f"{number(fit.get('image_rms_arcsec'))}, {number(fit.get('image_median_residual_arcsec'))}, {number(fit.get('image_max_residual_arcsec'))}"),
                                    ("chi-square / dof (catalog errors)", f"{number(fit.get('chi_square'), 1)} / {fit.get('dof', '-')} = {number(fit.get('reduced_chi_square'), 2)}")]),
        ("Sampler", [("chains x (warmup + samples)", f"{config.runtime.chains} x ({config.sampling.warmup} + {config.sampling.samples})"),
                     ("draws, divergences", f"{len(posterior.samples)}, {int(np.asarray(posterior.diverging).sum())}"),
                     ("mean accept, mean steps", f"{float(np.mean(posterior.accept_prob)):.3f}, {float(np.mean(posterior.num_steps)):.1f}"),
                     ("max R-hat, min ESS", f"{number(diagnostics.get('max_r_hat'))}, {number(diagnostics.get('min_n_eff'), 0)}"),
                     ("step size per chain", ", ".join(f"{c['step_size']:.3g}" for c in adaptation.get("per_chain", [])) or "-"),
                     ("E-BFMI per chain (< 0.3 is bad)", ", ".join(f"{c['e_bfmi']:.2f}" for c in adaptation.get("per_chain", [])) or "-"),
                     ("transitions at the tree-depth cap", ", ".join(f"{100 * c['tree_depth_saturation']:.0f}%" for c in adaptation.get("per_chain", [])) or "-"),
                     ("initialization", "WARNING, see the final log" if (diagnostics.get("start_check", {}).get("warnings")
                                                                      or diagnostics.get("mass_seed", {}).get("warning")) else "ok")]),
        ("Far-field audit", [("verdict", str(audit.get("approximation_validation", "-"))),
                             ("min R/r_cut over draws (guard)", f"{number(audit.get('min_separation_over_cut_radius'), 2)} ({audit.get('guard_cut_radii', '-')})"),
                             ("draws near / inside the guard", f"{100 * audit.get('fraction_of_draws_near_guard', 0.0):.0f}% / {audit.get('draws_inside_guard', 0)}")]),
        ("Time", [("init stage", _duration(timing.get("init"))), ("sampling", _duration(timing.get("sampling"))),
                  ("outputs", _duration(timing.get("outputs"))), ("total", _duration(timing.get("total")))]),
    ]
    table = Table(title="Run summary", title_justify="left", show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold"); table.add_column()
    for section, rows in sections:
        table.add_row(f"[underline]{section}[/underline]", "")
        for label, value in rows:
            table.add_row(f"  {label}", value)
        table.add_row("", "")
    console = Console(width=120, color_system=None, force_terminal=False)
    with console.capture() as capture:
        console.print(table)
    return capture.get().rstrip()


def search_summary_table(search: dict[str, Any]) -> str:
    """The init search, one row per stage: what was optimized, from how many starts, what came out and how long it took.
    Rendered with rich without colour."""
    from rich.console import Console
    from rich.table import Table
    stages = search.get("stages") or []
    table = Table(title="Init search", title_justify="left")
    for name, justify in (("stage", "left"), ("optimizer", "left"), ("parameters", "right"), ("starts", "right"), ("finite", "right"), ("modes", "right"),
                          ("within margin", "right"), ("carried", "right"), ("mean iterations", "right"), ("best log posterior", "right"), ("time", "right")):
        table.add_column(name, justify=justify, no_wrap=True)
    for row in stages:
        table.add_row(row["stage"], row["method"], str(row["active"]), str(row["starts"]), "-" if row.get("finite") is None else str(row["finite"]),
                      str(row["modes"]), str(row["within_margin"]), str(row.get("kept", "-")), f"{row['mean_iterations']:.0f}",
                      f"{row['best_log_posterior']:.2f}", _duration(row["seconds"]))
    modes = search.get("modes") or []
    best = modes[0]["log_posterior"] if modes else float("nan")
    footer = (f"best log posterior {best:.2f}; {sum(m['within_margin'] for m in modes)} mode(s) within the margin seed the chains; "
              f"total {_duration(search.get('total_seconds'))}")
    console = Console(width=170, color_system=None, force_terminal=False)
    with console.capture() as capture:
        console.print(table)
    return capture.get().rstrip() + "\n" + footer


def far_field_threshold_table(state: ModelState) -> str:
    """The far-field threshold study of the current classification as a table: for each R/r_cut, how many members
    would be exact and far-field and what deflection error the far-field members leave at the images; the configured
    threshold is marked.  Rendered with rich, without colour, so it reads the same in a log file."""
    from rich.console import Console
    from rich.table import Table
    d = state.members.classification
    rows = d.get("threshold_study") or []
    if not rows:
        return "far-field threshold study: no members"
    tolerance = d.get("tolerance")
    table = Table(title=f"Far-field threshold study ({d.get('member_count', 0)} members, {state.images.n_images} images; "
                        f"error = deflection left at the images by the far-field members at the current model)", title_justify="left")
    for name, justify in (("", "center"), ("R/r_cut", "right"), ("exact", "right"), ("far-field", "right"), ("max error (µas)", "right"),
                          ("median error (µas)", "right"), ("max whitened", "right"), (f"images > {tolerance:g} sigma" if tolerance else "images over tol.", "right"),
                          ("worst image", "left")):
        table.add_column(name, justify=justify)
    for row in rows:
        style = "bold" if row["selected"] else None
        table.add_row("*" if row["selected"] else "", f"{row['guard_cut_radii']:g}", str(row["exact"]), str(row["far_field"]),
                      f"{1000.0 * row['max_error_mas']:.2g}", f"{1000.0 * row['median_error_mas']:.2g}", f"{row['max_whitened_error']:.1e}",
                      str(row["images_over_tolerance"]), row["worst_image"] or "none", style=style)
    console = Console(width=140, color_system=None, force_terminal=False)
    with console.capture() as capture:
        console.print(table)
    return capture.get().rstrip()


def classification_report(state: ModelState, *, provisional: dict[str, Any] | None = None, top: int = 15) -> str:
    """Counts, why each member is exact, the highest-ranked members, and changes since the provisional pass."""
    d = state.members.classification
    rankings = d.get("rankings", [])
    promoted = {p["member"] for p in d.get("promotions", [])}
    by_truncation = sum(1 for r in rankings if r["exact"] and (r["truncation"] or 0) >= 1)
    by_ellipticity = sum(1 for r in rankings if r["exact"] and (r["truncation"] or 0) < 1 and (r["ellipticity"] or 0) >= 1)
    by_rank = sum(1 for r in rankings if r["exact"] and (r["score"] or 0) < 1 and r["member"] not in promoted)
    far = [r for r in rankings if not r["exact"]]
    lines = ["=== Member classification ==="]
    for potfile in state.potfiles:
        pivot = potfile.get("pivot")
        if pivot:
            lines.append(f"  pivot {potfile['id']}: mag0 = {pivot['mag0']:.0f} (leverage-weighted member magnitude {pivot['weighted_mean_mag']:.2f}, "
                         f"median {pivot['member_median_mag']:.2f}); sigma and cutkpc priors moved from mag {pivot['prior_mag0']:.2f} "
                         f"by x{pivot['sigma_factor']:.3f} and x{pivot['cutkpc_factor']:.3f}: sigma* {potfile['sigma_nominal']:.1f} km/s, "
                         f"r_cut* {potfile['cutkpc_nominal']:.1f} kpc")
    if d.get("dropped_members"):
        lines.append(f"  impact cut: {len(d['dropped_members'])} catalog members dropped from the model (deflection at every image "
                     f"below {d.get('min_deflection_arcsec'):g}\" at the initial model)")
    lines += [
             f"  members {d['member_count']}: exact {d['exact_count']} (mandatory {d['mandatory_count']}), far-field {d['member_count'] - d['exact_count']}; "
             f"deviations: {state.packed_lens_spec.deviation_mode}, centered for {d.get('centered_count', 0)} members",
             f"  exact because an image lies within {d['guard_cut_radii']:g} r_cut: {by_truncation}; because of ellipticity error > {d['tolerance']:g}: {by_ellipticity}; "
             f"promoted by the collective budget: {len(promoted)}; added by top_exact: {by_rank}",
             f"  cut radii for classification: reference {d.get('reference_cut_kpc', float('nan')):.1f} kpc x {d.get('inflated_cut_kpc', 0) / max(d.get('reference_cut_kpc', 1), 1e-12):.1f} = "
             f"{d.get('inflated_cut_kpc', float('nan')):.1f} kpc, beta in [{d.get('beta_range', [d.get('beta_radius', float('nan'))] * 2)[0]:.2f}, {d.get('beta_range', [d.get('beta_radius', float('nan'))] * 2)[1]:.2f}]"
             + (f" (prior {d['prior_quantile']:.3f} quantiles or the current model, whichever is larger)" if d.get('prior_quantile') else "")]
    if far:
        worst = max(far, key=lambda r: r["score"] or 0)
        min_ratio = min(r["min_separation_arcsec"] / r["cut_radius_arcsec"] for r in far if r["cut_radius_arcsec"] > 0)
        if d.get("guard_headroom") is not None:
            lines.append(f"  guard headroom: r_cut* can grow x{d['guard_headroom']:.2f} before {d.get('guard_headroom_member')} reaches the guard"
                         + ("  <-- WARNING: the sampler will be pinned; raise members.cut_radius_inflation" if d["guard_headroom"] < 2.0 else ""))
        lines.append(f"  far-field: highest score {worst['score']:.3g} ({worst['member']}), smallest R/r_cut {min_ratio:.2f} (guard {d['guard_cut_radii']:g}), "
                     f"summed ellipticity budget per image < {d['tolerance']:g}")
    lines += _rank_table(rankings[:top], f"  top {min(top, len(rankings))} members by score:")
    if far:
        lines += _rank_table(far[:5], "  far-field members closest to the threshold:")
    if provisional is not None:
        before = {r["member"] for r in provisional.get("rankings", []) if r["exact"]}
        after = {r["member"] for r in rankings if r["exact"]}
        lines.append(f"  change since the provisional pass: +{len(after - before)} exact, -{len(before - after)} exact "
                     f"(provisional {len(before)} -> final {len(after)})")
        if after - before:
            lines.append("    made exact: " + ", ".join(sorted(after - before)[:20]) + (" ..." if len(after - before) > 20 else ""))
        if before - after:
            lines.append("    made far-field: " + ", ".join(sorted(before - after)[:20]) + (" ..." if len(before - after) > 20 else ""))
    return "\n".join(lines)


def audit_report(state: ModelState, audit: dict[str, Any], *, posterior_summary: dict[str, Any] | None = None) -> str:
    """Guard margins over the draws, the far-field error at the median, and the median-model reclassification."""
    lines = ["=== Posterior audit of the far-field approximation ===",
             f"  draws {audit.get('draws')}, far-field members {audit.get('far_member_count')}, exact {audit.get('exact_member_count')}"]
    if "min_separation_over_cut_radius" in audit:
        lines.append(f"  guard: min R/r_cut over draws {audit['min_separation_over_cut_radius']:.2f} (limit {audit['guard_cut_radii']:g}), "
                     f"draws inside the guard {audit['draws_inside_guard']}, worst member {audit['worst_member']}, "
                     f"draws within 10% of the guard {100 * audit.get('fraction_of_draws_near_guard', 0.0):.0f}%")
        if audit.get("guard_pinned"):
            lines.append("  WARNING: the posterior is pinned against the far-field guard; the cut-radius relation was shaped by the guard, "
                         "not by the data.  Raise members.cut_radius_inflation (classify with a larger cut radius) and rerun.")
    w = audit.get("whitened_far_field_error_median_model", {})
    if w:
        lines.append(f"  whitened far-field error at the median model: max {w['max']:.3g} (image {w['worst_image']}), median {w['median']:.3g}, "
                     f"tolerance {audit['tolerance']:g}")
    r = audit.get("reclassification_at_median")
    if r:
        lines.append(f"  reclassifying at the posterior median: exact {r['exact_count']} vs {r['current_exact_count']} now; "
                     f"would add {len(r['would_add'])}, remove {len(r['would_remove'])}")
        if r["would_add"]:
            lines.append("    would add: " + ", ".join(r["would_add"][:20]) + (" ..." if len(r["would_add"]) > 20 else ""))
        if r["would_remove"]:
            lines.append("    would remove: " + ", ".join(r["would_remove"][:20]) + (" ..." if len(r["would_remove"]) > 20 else ""))
    if posterior_summary:
        lines.append("  sampler: " + ", ".join(f"{k} {v}" for k, v in posterior_summary.items()))
    lines.append(f"  far-field approximation verdict: {audit.get('approximation_validation')} (sampler and image recovery: see the quality gates)")
    return "\n".join(lines)


# Start-parameter report: the global parameters at a model, grouped by what they describe, with their priors.
_START_GROUPS = (("cluster", "Cluster-scale halos"), ("shear", "External shear"), ("independent", "Independent member halos"),
                 ("scaling", "Member scaling relation"), ("error", "Error model (GRF + GP)"))
_BOUND_MARGIN = 0.02            # flag a value within this fraction of its prior range (in sampling coordinates) of a bound
# Status of a value in its prior: red within 2% of a bound or |z| >= 3, yellow within 10% of a bound or |z| >= 2, else green.
_RED_EDGE, _YELLOW_EDGE, _RED_Z, _YELLOW_Z = 0.02, 0.10, 3.0, 2.0
_STATUS_STYLE = {"green": "green", "yellow": "yellow", "red": "bold red"}


def _num(value: float) -> str:
    if value is None or not np.isfinite(value):
        return ""
    value = float(value)
    if abs(value) < 1e-12:
        return "0"
    if abs(value) >= 1e4 or abs(value) < 1e-3:
        return f"{value:.3g}"
    return f"{value:.4g}"


def _prior_text(spec) -> str:
    """The prior in physical units, as the sampler applies it."""
    log = spec.transform_kind == "log_positive"
    lo, hi = float(spec.lower), float(spec.upper)
    bounds = ""
    if np.isfinite(lo) or np.isfinite(hi):
        plo = spec.physical_lower if spec.physical_lower is not None else (np.exp(lo) if log and np.isfinite(lo) else lo)
        phi = spec.physical_upper if spec.physical_upper is not None else (np.exp(hi) if log and np.isfinite(hi) else hi)
        bounds = f" on [{_num(plo) or '-inf'}, {_num(phi) or 'inf'}]"
    if spec.prior_kind == "uniform":
        return f"{'log-uniform' if log else 'uniform'}{bounds}"
    if spec.prior_kind == "student_t":
        return f"student-t(dof {_num(spec.dof)}, {_num(spec.mean)}, {_num(spec.std)})"
    if spec.prior_kind in ("normal", "truncated_normal"):
        mean, std = float(spec.mean), float(spec.std)
        core = f"log-normal(median {_num(np.exp(mean))}, ln-width {_num(std)})" if log else f"normal({_num(mean)}, {_num(std)})"
        return core + (bounds if spec.prior_kind == "truncated_normal" else "")
    return spec.prior_kind


def _prior_position(spec, latent: float) -> str:
    """Where the value sits in its prior: z for normal priors, the fraction of the range for uniform ones, and a flag
    within 2% of a bound."""
    lo, hi = float(spec.lower), float(spec.upper)
    parts = []
    if spec.prior_kind in ("normal", "truncated_normal") and spec.std:
        parts.append(f"z {((latent - float(spec.mean)) / float(spec.std)):+.1f}")
    elif np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        parts.append(f"{100.0 * (latent - lo) / (hi - lo):.0f}% of range")
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        if latent - lo <= _BOUND_MARGIN * (hi - lo):
            parts.append("AT LOWER BOUND")
        elif hi - latent <= _BOUND_MARGIN * (hi - lo):
            parts.append("AT UPPER BOUND")
    return ", ".join(parts)


def _prior_status(spec, latent: float) -> str:
    """green, yellow or red: how close the value sits to a prior bound (fraction of the range in sampling coordinates)
    or how far from a normal prior's centre (z)."""
    lo, hi = float(spec.lower), float(spec.upper)
    edge = np.inf
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:               # a one-sided bound (a half-normal) is judged by z alone
        edge = min(latent - lo, hi - latent) / (hi - lo)
    z = abs((latent - float(spec.mean)) / float(spec.std)) if spec.prior_kind in ("normal", "truncated_normal") and spec.std else 0.0
    if edge <= _RED_EDGE or z >= _RED_Z:
        return "red"
    if edge <= _YELLOW_EDGE or z >= _YELLOW_Z:
        return "yellow"
    return "green"


def _start_group(spec, independent_ids: set[str]) -> str | None:
    family = spec.component_family
    if family == "error_scale":
        return "error"
    if family in ("scaling", "member_velocity_scatter", "softening_length"):
        return "scaling"
    if family == "large":
        if spec.potential_id == "shear" or spec.field in ("gamma1", "gamma2"):
            return "shear"
        return "independent" if spec.potential_id in independent_ids else "cluster"
    return None                                     # member offsets and sources: summarized, not listed


def start_parameter_report(state: ModelState, latent: np.ndarray, physical: np.ndarray, *, independent_ids=(),
                           title: str = "at the start", color: bool = False) -> str:
    """The global parameters at one model (``latent`` sampling coordinates and their ``physical`` values), one table
    per group: cluster-scale halos, external shear, independent member halos, the member scaling relation (with its
    scatters) and the error model, each with its prior and where the value sits in it.  The member offsets and the
    source positions are summarized below the tables.  Each row has a status, green, yellow or red (see _prior_status);
    with ``color`` the status and the position in the prior are coloured for a terminal, otherwise the text is plain for
    a log file.  Rendered with rich."""
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    latent, physical = np.asarray(latent, dtype=float), np.asarray(physical, dtype=float)
    independent_ids = {str(i) for i in independent_ids}
    rows: dict[str, list] = {key: [] for key, _ in _START_GROUPS}
    offsets: dict[str, list[tuple[str, float]]] = {}
    n_sources = 0
    for index, spec in enumerate(state.parameter_specs):
        group = _start_group(spec, independent_ids)
        if group is not None:
            rows[group].append((spec, latent[index], physical[index]))
        elif spec.component_family in ("member_scatter_offset", "member_shape_offset"):
            offsets.setdefault(spec.field, []).append((spec.name.split(".")[1] if spec.name.count(".") >= 2 else spec.name, float(latent[index])))
        elif spec.component_family == "source_position":
            n_sources += 1
    console = (Console(width=160, color_system="standard", force_terminal=True) if color
               else Console(width=160, color_system=None, force_terminal=False))
    counts = {"green": 0, "yellow": 0, "red": 0}
    with console.capture() as capture:
        for key, label in _START_GROUPS:
            if not rows[key]:
                continue
            table = Table(title=f"{label} {title}", title_justify="left")
            for name, justify in (("component", "left"), ("parameter", "left"), ("value", "right"), ("prior", "left"), ("in prior", "left"),
                                  ("status", "left")):
                table.add_column(name, justify=justify, no_wrap=True)
            previous = None
            for spec, z, value in rows[key]:
                component = spec.potential_id if spec.potential_id != previous else ""
                if previous is not None and spec.potential_id != previous:
                    table.add_section()
                previous = spec.potential_id
                status = _prior_status(spec, z)
                counts[status] += 1
                style = _STATUS_STYLE[status]
                table.add_row(component, spec.field, _num(value), _prior_text(spec), Text(_prior_position(spec, z), style=style),
                              Text(status, style=style))
            console.print(table)
    lines = [capture.get().rstrip(),
             f"status: {counts['green']} green, {counts['yellow']} yellow, {counts['red']} red (red: within 2% of a prior bound or "
             f"3 prior widths from its centre; the run continues)"]
    names = {"velocity_offset": "velocity", "radius_offset": "radius", "shape_e1_offset": "shape e1", "shape_e2_offset": "shape e2"}
    offset_prior = {s.field: s for s in state.parameter_specs if s.component_family in ("member_scatter_offset", "member_shape_offset")}
    for field, values in offsets.items():
        array = np.asarray([v for _, v in values])
        worst = int(np.argmax(np.abs(array)))
        prior = _prior_text(offset_prior[field]) if field in offset_prior else "N(0, 1)"
        lines.append(f"member {names.get(field, field)} offsets (unit, prior {prior}): {len(array)} members, rms {np.sqrt(np.mean(array ** 2)):.2f}, "
                     f"max |offset| {abs(array[worst]):.2f} (member {values[worst][0]})")
    cap = np.asarray(getattr(state.packed_lens_spec, "ellipticity_soft_cap", []), dtype=float)
    width = np.asarray(getattr(state.packed_lens_spec, "ellipticity_soft_width", []), dtype=float)
    spec_names = [s.name for s in state.parameter_specs]
    for c in np.flatnonzero(np.isfinite(cap)) if cap.size else []:
        halo = str(state.base_components[int(c)]["id"])
        e = [physical[spec_names.index(f"{halo}.{k}")] if f"{halo}.{k}" in spec_names else np.nan for k in ("e1", "e2")]
        modulus = float(np.hypot(*e))
        lines.append(f"halo {halo}: smooth ellipticity cap from |e| = {cap[c]:.3f} (width {width[c]:g}); |e| = {modulus:.3f} here, "
                     f"log prior {-np.logaddexp(0.0, (modulus - cap[c]) / width[c]):.2f}")
    if n_sources:
        lines.append(f"source positions: {n_sources // 2} sources (not listed)")
    return "\n".join(lines)
