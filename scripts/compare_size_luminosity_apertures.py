"""Exploratory luminosity-scaled aperture masses; no changes to inference.

Sizes are assumed, not measured. Ranking normalizations against truth is tuning.
Exact full-model mass uses the deflection boundary integral (divergence theorem).
Truth uses area quadrature of the native FITS bilinear interpolant.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from scipy.spatial import ConvexHull
from astropy.wcs.utils import proj_plane_pixel_area
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from types import SimpleNamespace
from lenscluster import artifacts
from lenscluster.evaluator import Evaluator
from lenscluster.state import build_state, with_member_classes
from lenscluster.truth_recovery import load_truth_fits, mass_per_kappa_pixel_msun
from lenscluster.plot_style import PAPER_STYLE


def polar_rule(nr, nt):
    q, w = np.polynomial.legendre.leggauss(nr)
    r = (q + 1) / 2
    phi = (np.arange(nt) + .5) * 2 * np.pi / nt
    unit = r[:, None, None] * np.stack((np.cos(phi), np.sin(phi)), -1)[None]
    weight = np.broadcast_to((w * r / 2)[:, None], (nr, nt)) * 2 * np.pi / nt
    np.testing.assert_allclose(weight.sum(), np.pi, rtol=1e-14)
    return unit.reshape(-1, 2), weight.ravel()


def write_report(out, summary, provenance):
    all_best = summary[summary['sample']=='all'].sort_values('median_absolute_fractional_error').iloc[0]
    hull_best = summary[summary['sample']=='image_hull'].sort_values('median_absolute_fractional_error').iloc[0]
    columns = ['normalization_kpc', 'sample', 'n', 'median_ratio',
               'median_absolute_fractional_error', 'fraction_within_20_percent']
    table = '| ' + ' | '.join(columns) + ' |\n'
    table += '| ' + ' | '.join(['---']*len(columns)) + ' |\n'
    for row in summary[columns].itertuples(index=False, name=None):
        table += '| ' + ' | '.join(f'{v:.5g}' if isinstance(v, float) else str(v) for v in row) + ' |\n'
    text = ['# Hera luminosity-scaled aperture mass comparison', '',
        'Re = R20 * 10^[-0.24 (m_F160W - 20)] kpc; circular aperture radius = 2 Re.',
        'The assumed slope 0.6 approximates the early-type luminosity-size slope 0.63 of Bernardi et al. (2003): https://arxiv.org/abs/astro-ph/0301624 . This is not a calibrated F160W relation at the Hera redshift.', '', table,
        f'Best across all members: R20 = {all_best.normalization_kpc:g} kpc, median absolute fractional error {100*all_best.median_absolute_fractional_error:.2f}%.',
        f'Best inside the image hull: R20 = {hull_best.normalization_kpc:g} kpc, median absolute fractional error {100*hull_best.median_absolute_fractional_error:.2f}%.', '',
        'These are projected total masses, including cluster background and neighbors, not isolated bound subhalo masses.',
        'No size scatter was injected. Larger apertures can look better because smooth cluster mass dominates. Best means best among the tested values, tuned against truth, not a measurement of the galaxy sizes.',
        'No Ares run exists under results/nov6_gammafree. No posterior aperture uncertainty is claimed. The conditional GRF convergence correction is not included.',
        'Model masses use exact-kernel deflection boundary integrals. Truth masses use area quadrature of the native bilinear-interpolated convergence map.',
        'Quadrature convergence does not remove finite truth-map resolution (0.110916 arcsec per pixel). Smallest apertures are only a few pixels across.', '',
        '## Provenance', '```json', json.dumps(provenance, indent=2), '```', '',
        '## Files', '- normalization_comparison.pdf: error versus normalization and mass-ratio cumulative distributions.',
        '- best_normalization_recovery.pdf: mass recovery and magnitude dependence at the all-member best normalization.',
        '- member_aperture_masses.csv: sizes, masses, ratios, sample masks, and numerical integration errors per galaxy.',
        '- normalization_comparison.csv: aggregate metrics for both samples.',
        '- aperture_arrays.npz: full mass arrays, identities, masks, radii, and the physical median parameter vector.',
        '- provenance.json: assumptions, run identity, timing, input hash, and numerical precision.',
        '- quadrature_validation.json: independent analytic area versus boundary integration checks (when run).']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output; out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    import jax
    jax.config.update('jax_enable_x64', True)
    print('Loading saved configuration and member classes', flush=True)
    init, final = a.run/'init/artifacts', a.run/'final'
    config = artifacts.load_config(final/'artifacts')
    classes = json.loads((init/artifacts.CLASSIFICATION_FILE).read_text())
    state = build_state(config, drop_member_ids=classes.get('dropped_members', ()))
    centered = set(classes.get('centered_members', []))
    state = with_member_classes(state, artifacts.load_exact_positions(init), classes,
        centered=np.array([m in centered for m in state.members.member_ids]))
    saved = json.loads((init/artifacts.STATE_FILE).read_text())
    assert [s.name for s in state.parameter_specs] == [s['name'] for s in saved['parameter_specs']]
    assert list(state.members.member_ids) == saved['member_ids']
    posterior = artifacts.load_posterior(final/'artifacts')
    print('Building field evaluator; physical saved parameters used directly', flush=True)
    evaluator = Evaluator(state, config.likelihood.likelihood_mode)
    paths = SimpleNamespace(tables=final/'tables')
    members = pd.read_csv(paths.tables / 'member_diagnostics.csv', dtype={'member_id': str, 'catalog_id': str})
    assert members.member_id.tolist() == list(state.members.member_ids)
    images = pd.read_csv(paths.tables / 'diagnostic_image_index.csv')
    centers = members[['x_centre', 'y_centre']].to_numpy()
    hull = ConvexHull(images[['x_obs_arcsec', 'y_obs_arcsec']].to_numpy())
    inside = np.all(centers @ hull.equations[:, :2].T + hull.equations[:, 2] <= 0, axis=1)
    normalizations = np.array([1., 2., 3., 4., 5., 6.])
    pivot, slope = 20., .6
    sizes = normalizations[:, None] * 10 ** (-.4 * slope * (members.catalog_mag.to_numpy()[None] - pivot))
    radii = 2 * sizes / state.kpc_per_arcsec
    native, wcs = load_truth_fits(config.truth.kappa_true_fits)
    z = float(config.truth.caustic_source_redshift)
    pixel_area = float(proj_plane_pixel_area(wcs)) * 3600 ** 2
    mass_factor = mass_per_kappa_pixel_msun(state, wcs, z) / pixel_area
    _, ra0, dec0 = state.reference

    def pixels(xy):
        return wcs.world_to_pixel_values(ra0 - xy[..., 0] / (3600 * np.cos(np.deg2rad(dec0))),
                                         dec0 + xy[..., 1] / 3600)

    # Common sample at every normalization; never silently integrate clipped apertures.
    phi = np.arange(1024) * 2 * np.pi / 1024
    directions = np.stack((np.cos(phi), np.sin(phi)), -1)
    edges = centers[:, None] + radii[-1, :, None, None] * directions
    px, py = pixels(edges)
    valid = np.all((px >= 1) & (py >= 1) & (px <= native.shape[1]-2) & (py <= native.shape[0]-2), axis=1)
    assert np.isfinite(members.catalog_mag).all()
    physical = np.median(posterior.physical, axis=0)
    packed = evaluator.packed_state_for_z(jnp.asarray(physical), z)

    def model_integrals(nt, packed_model):
        phi = (np.arange(nt)+.5)*2*np.pi/nt
        n = np.stack((np.cos(phi), np.sin(phi)), -1)
        result = np.zeros_like(radii)
        for k, r in enumerate(radii):
            xy = centers[:, None] + r[:, None, None]*n
            alpha, _ = evaluator.total_fields_at(xy.reshape(-1,2), packed_model, chunk=4096)
            result[k] = np.pi*r*np.mean(np.sum(alpha.reshape(len(centers),nt,2)*n, axis=-1), axis=1)
        return result

    def truth_integrals(nr, nt):
        unit, weight = polar_rule(nr, nt)
        result = np.full_like(radii, np.nan)
        for k, rs in enumerate(radii):
            for g in np.flatnonzero(valid):
                xy = centers[g] + rs[g]*unit
                px, py = pixels(xy)
                values = map_coordinates(native, [py,px], order=1, mode='constant', cval=np.nan)
                result[k,g] = rs[g]**2 * (values @ weight)
        return result

    print(f'{len(members)} members; {valid.sum()} complete apertures; {(valid & inside).sum()} inside image hull', flush=True)
    print('Exact median model boundary integration, 256 and 512 angles', flush=True)
    model_coarse = model_integrals(256, packed)
    model = model_integrals(512, packed)
    print('Native truth integration, 48x192 and 96x384 points per aperture', flush=True)
    truth_coarse = truth_integrals(48,192)
    truth = truth_integrals(96,384)
    model_error = np.abs(model/model_coarse-1)
    truth_error = np.abs(truth/truth_coarse-1)
    # Increase resolution if the tested quadrature differs by more than 0.5%.
    if np.nanmax(model_error[:,valid]) > .005:
        old=model; model=model_integrals(1024,packed); model_error=np.abs(model/old-1)
    if np.nanmax(truth_error[:,valid]) > .005:
        old=truth; truth=truth_integrals(192,768); truth_error=np.abs(truth/old-1)
    ratio = model/truth
    rows=[]; summaries=[]
    for k, norm in enumerate(normalizations):
        for g, m in enumerate(members.itertuples()):
            rows.append(dict(member_id=m.member_id,catalog_id=m.catalog_id,mag=m.catalog_mag,
                normalization_kpc=norm,re_kpc=sizes[k,g],aperture_arcsec=radii[k,g],
                complete_aperture=bool(valid[g]),inside_image_hull=bool(inside[g]),
                truth_mass_msun=truth[k,g]*mass_factor,model_mass_msun=model[k,g]*mass_factor,
                ratio=ratio[k,g],model_quadrature_error=model_error[k,g],truth_quadrature_error=truth_error[k,g]))
        for label,mask in [('all',valid),('image_hull',valid & inside)]:
            r=ratio[k,mask]; q=np.quantile(r,[.16,.5,.84])
            summaries.append(dict(normalization_kpc=norm,sample=label,n=int(mask.sum()),
                median_ratio=q[1],p16_ratio=q[0],p84_ratio=q[2],
                median_absolute_fractional_error=float(np.median(abs(r-1))),
                fraction_within_20_percent=float(np.mean(abs(r-1)<.2)),
                median_aperture_arcsec=float(np.median(radii[k,mask]))))
    table=pd.DataFrame(rows); summary=pd.DataFrame(summaries)
    table.to_csv(out/'member_aperture_masses.csv',index=False)
    summary.to_csv(out/'normalization_comparison.csv',index=False)
    np.savez_compressed(out/'aperture_arrays.npz',model_mass=model*mass_factor,truth_mass=truth*mass_factor,
        normalizations_kpc=normalizations,radii_arcsec=radii,member_ids=members.member_id.to_numpy(dtype=str),
        complete_aperture=valid,inside_image_hull=inside,median_physical=physical)
    print(summary.to_string(index=False),flush=True)
    plt.style.use(Path(__file__).resolve().parents[1]/'src/lenscluster/style.mplstyle')
    plt.rcParams.update(PAPER_STYLE)
    fig,axs=plt.subplots(1,2,figsize=(7.1,3.2),layout='constrained')
    for label,color,text in [('all','tab:blue','All members'),('image_hull','tab:orange','Inside image hull')]:
        s=summary[summary['sample']==label]
        axs[0].plot(s.normalization_kpc,100*s.median_absolute_fractional_error,'o-',color=color,label=text,ms=4)
    axs[0].set(xlabel=r'$R_{e,20}\ [\mathrm{kpc}]$',ylabel='Median absolute mass error [%]')
    axs[0].legend(frameon=False)
    for k,norm in enumerate(normalizations):
        axs[1].plot(np.sort(ratio[k,valid]),np.arange(1,valid.sum()+1)/valid.sum(),label=f'{norm:g} kpc')
    axs[1].axvline(1,color='.4',ls='--',lw=.8)
    axs[1].set(xlabel=r'$M_{\rm model}/M_{\rm truth}$',ylabel='Cumulative fraction',xlim=(0,2.5),ylim=(0,1))
    axs[1].legend(frameon=False,fontsize=7)
    fig.savefig(out/'normalization_comparison.pdf',bbox_inches='tight');plt.close(fig)
    best=summary[summary['sample']=='all'].sort_values('median_absolute_fractional_error').iloc[0]
    k=int(np.flatnonzero(normalizations==best.normalization_kpc)[0])
    fig,axs=plt.subplots(1,2,figsize=(7.1,3.2),layout='constrained')
    for mask,color,label in [(valid & ~inside,'tab:blue','Outside image hull'),(valid & inside,'tab:orange','Inside image hull')]:
        axs[0].scatter(truth[k,mask]*mass_factor,model[k,mask]*mass_factor,s=10,color=color,label=label)
        axs[1].scatter(members.catalog_mag.to_numpy()[mask],ratio[k,mask],s=10,color=color)
    limits=[np.nanmin(truth[k,valid]*mass_factor)*.7,np.nanmax(truth[k,valid]*mass_factor)*1.3]
    axs[0].plot(limits,limits,color='.4',ls='--',lw=.8)
    axs[0].set(xscale='log',yscale='log',xlabel=r'$M_{\rm truth}\ [M_\odot]$',ylabel=r'$M_{\rm model}\ [M_\odot]$')
    axs[0].legend(frameon=False,fontsize=7)
    axs[1].axhline(1,color='.4',ls='--',lw=.8)
    axs[1].set(xlabel=r'$m_{\rm F160W}$',ylabel=r'$M_{\rm model}/M_{\rm truth}$')
    fig.savefig(out/'best_normalization_recovery.pdf',bbox_inches='tight');plt.close(fig)
    provenance=dict(run=str(a.run.resolve()),model='componentwise physical posterior median; exact parametric kernels',
        grf='Not added: matches existing truth_recovery lens_quantities full parametric mass map; GRF is a conditional stochastic field.',
        equation='Re[kpc] = R20 * 10**(-0.24*(m_F160W-20)); aperture radius = 2 Re',
        slope=.6,pivot_magnitude=20,normalizations_kpc=normalizations.tolist(),z_source=z,
        selection='All saved population members, common complete-aperture mask; center inside convex hull reported separately',
        metric='Median absolute fractional mass error; normalization selected against truth, not independently validated',
        model_quadrature_max=float(np.nanmax(model_error[:,valid])),truth_quadrature_max=float(np.nanmax(truth_error[:,valid])),
        truth_pixel_arcsec=float(np.sqrt(pixel_area)),mass_factor_msun_per_kappa_arcsec2=mass_factor,
        elapsed_seconds=time.perf_counter()-started,
        posterior_sha256=hashlib.sha256((a.run/'final/artifacts/posterior.npz').read_bytes()).hexdigest())
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    write_report(out, summary, provenance)


if __name__=='__main__':
    main()
