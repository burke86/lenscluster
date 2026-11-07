"""Geometric diagnostics of cached M0416 curves; not microlensing event rates.

Run with the existing lenscluster interpreter. No fitting or cache mutation.
"""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM

DEFAULT = 'results/nov3o_densemass_notfast/m0416_canucs2025/w500s250t8_gamma4_allblocks/final'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default=DEFAULT)
    parser.add_argument('--output', default='output/pdf/m0416_dragon_grf')
    args = parser.parse_args()
    run, out = Path(args.run), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    path = run/'tables/fold_pairs_source_caustics.npz'
    meta = json.loads(path.with_suffix('.json').read_text())
    arrays = np.load(path)
    scale = float(FlatLambdaCDM(H0=70, Om0=.3).kpc_proper_per_arcmin(.396).value/60)
    plt.rcParams.update({'font.family':'serif', 'mathtext.fontset':'stix', 'font.size':10,
                         'axes.grid':False, 'pdf.fonttype':42})
    figures = [plt.subplots(3, 3, figsize=(11, 9), layout='constrained') for _ in range(3)]
    summary, samples, profiles = [], {}, []
    for i, sm in enumerate(meta['systems']):
        s = {k[len(f'system_{i}_'):]:arrays[k] for k in arrays.files if k.startswith(f'system_{i}_')}
        normal = s['normal']; tangent = np.array([normal[1], -normal[0]])
        along = (s['seeds']-s['origin']) @ tangent
        # Use the central half of the displayed field: a fixed geometric aperture,
        # independent of how many knots happen to have been catalogued.
        mask = (abs(along) <= .25*sm['fov']) & (s['grf_band_valid_draws'] >= .8*len(s['critical']))
        p, g = s['critical'][:, mask], s['critical_grf_draws'][:, mask]
        delta = ((g-p) @ normal)*scale
        good = np.isfinite(delta).all(axis=1)
        delta = delta[good]
        if not len(delta) or delta.shape[1] < 3:
            raise ValueError(f'Insufficient complete paired curves for {sm["system_id"]}')
        # Every retained cross-section has equal weight; each complete draw has equal weight.
        shift = delta.mean(axis=1)
        residual = delta-shift[:, None]
        design = np.polynomial.polynomial.polyvander(along[mask], 2)
        detrended = delta-(design @ np.linalg.lstsq(design, delta.T, rcond=None)[0]).T
        best_trend = design @ np.linalg.lstsq(design, ((g[meta['best_draw']]-p[meta['best_draw']]) @ normal)*scale, rcond=None)[0]
        width = np.quantile(delta,.95,axis=1)-np.quantile(delta,.05,axis=1)
        best = int(meta['best_draw'])
        best_delta = ((g[best]-p[best]) @ normal)*scale
        record = dict(system=sm['system_id'], z_source=sm['z_source'], draws=len(delta),
                      cross_sections=int(mask.sum()), aperture_half_length_arcsec=.25*sm['fov'],
                      mean_shift_kpc=float(shift.mean()), shift_draw_q16_q50_q84_kpc=np.quantile(shift,[.16,.5,.84]).tolist(),
                      spatial_width90_draw_q16_q50_q84_kpc=np.quantile(width,[.16,.5,.84]).tolist(),
                      best_mean_shift_kpc=float(np.nanmean(best_delta)),
                      best_spatial_width90_kpc=float(np.nanquantile(best_delta,.95)-np.nanquantile(best_delta,.05)),
                      best_quadratic_residual_width90_kpc=float(np.nanquantile(best_delta-best_trend,.95)-np.nanquantile(best_delta-best_trend,.05)),
                      pooled_shift_q05_q95_kpc=np.quantile(delta,[.05,.95]).tolist(),
                      negative_reference_direction_fraction=float(np.mean(delta<0)))
        summary.append(record)
        samples[sm['system_id']+'_offset_kpc']=delta
        samples[sm['system_id']+'_quadratic_residual_kpc']=detrended
        samples[sm['system_id']+'_draw_indices']=np.asarray(meta['draw_indices'])[good]
        profiles.append((sm, along[mask], best_delta, s))
        for n, values in enumerate([delta.ravel(), residual.ravel(), shift]):
            ax=figures[n][1].flat[i]
            ax.hist(values, bins=30 if n<2 else 20, density=True, histtype='step', lw=1.5, color='#395c85')
            if n==1:
                ax.hist(detrended.ravel(),bins=30,density=True,histtype='step',lw=1,color='#c78330')
            ax.axvline(0,color='.55',lw=.7)
            if n==2:
                ax.axvspan(-.9,-.5,color='#c78330',alpha=.17)
                ax.axvline(-.7,color='#c78330',ls='--',lw=1)
            ax.text(.04,.94,sm['system_id'],transform=ax.transAxes,va='top',bbox={'facecolor':'white','edgecolor':'none','alpha':.8})
            ax.set_xlabel(['Paired curve displacement (kpc)', 'Displacement minus draw mean (kpc)', 'Mean curve displacement per draw (kpc)'][n])
            ax.set_ylabel('Normalized frequency')
    names=['paired_displacements','centered_wiggles','mean_shifts']
    for (fig,_),name in zip(figures,names):
        fig.get_layout_engine().set(rect=(0,.045,1,.955))
        note = {'paired_displacements':'Fixed reference-normal offsets; conditional-mean GRF; unconverged draws; not event counts.',
                'centered_wiggles':'Blue: mean removed. Orange: quadratic trend removed. Geometric residuals; not event counts.',
                'mean_shifts':'Orange: Dragon Arc event mean (-0.7 +/- 0.2 kpc), scale reference only; different observable.'}[name]
        fig.text(.5,.012,note,ha='center',fontsize=9)
        fig.savefig(out/(name+'.pdf')); plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(11,9),layout='constrained')
    for ax,(sm,x,d,s) in zip(axes.flat,profiles):
        ax.plot(x,d,color='#395c85',lw=1)
        ax.axhline(0,color='.55',lw=.7)
        ax.text(.04,.94,sm['system_id'],transform=ax.transAxes,va='top',bbox={'facecolor':'white','edgecolor':'none','alpha':.8})
        ax.set_xlabel('Reference tangent coordinate (arcsec)')
        ax.set_ylabel('Best-draw displacement (kpc)')
    fig.get_layout_engine().set(rect=(0,.045,1,.955))
    fig.text(.5,.012,'Same-draw GRF minus parametric curve; fixed reference normal; conditional-mean field only.',ha='center',fontsize=9)
    fig.savefig(out/'displacement_profiles.pdf'); plt.close(fig)
    np.savez_compressed(out/'histogram_samples.npz',**samples)
    result=dict(input=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        lens_redshift=.396, H0=70, Om0=.3, kpc_per_arcsec=scale,
        definition='(conditional-mean GRF curve - same-draw parametric curve) dot fixed reference normal; equal cross-section weights, equal complete-draw weights',
        sign='normal is grad(det A) at the reference anchor; not a verified parity assignment over the entire arc',
        aperture='abs(reference tangent coordinate) <= 0.25 * saved fov; >=80% available GRF draws per cross-section; then complete paired draws only',
        limitations=['Unconverged chain; draw quantiles are descriptive, not credible intervals',
          'Conditional means only; no residual latent GRF realizations',
          'Tracks nearest branch only; not all critical loops or arc-length weighting',
          'Automatic derivatives of linearly interpolated covariance can introduce derivative jumps',
          'Spatial 5-95% widths differ from the paper full event-band width',
          'No microlensing, luminosity function, source density, completeness, or magnification bias'],
        systems=summary)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
