"""Compare cached aperture normalizations using the Meneghetti et al. histogram statistics."""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from lenscluster.plot_style import PAPER_STYLE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--normalization', type=float, default=6., help='R_e at F160W=20, in kpc, for the overlaid histogram')
    args = parser.parse_args()
    data = pd.read_csv(args.directory/'member_aperture_masses.csv')
    data = data[data.complete_aperture]
    plt.style.use(Path(__file__).resolve().parents[1]/'src/lenscluster/style.mplstyle')
    plt.rcParams.update(PAPER_STYLE)
    selected = data[np.isclose(data.normalization_kpc, args.normalization)]
    if selected.empty:
        raise ValueError(f'No saved masses for normalization {args.normalization:g} kpc')
    hull = selected[selected.inside_image_hull]
    from lenscluster.aperture_mass_histogram import render_histogram
    counts = render_histogram(selected, args.directory/'all_vs_image_footprint_histogram.pdf')
    counts.to_csv(args.directory/'all_vs_image_footprint_histogram.csv', index=False)
    display_limits = np.quantile(selected.ratio, [.005, .995])
    (args.directory/'all_vs_image_footprint_histogram.md').write_text(
        f'Normalization: R_e = {args.normalization:g} * 10^[-0.24 (m_F160W - 20)] kpc; aperture radius = 2 R_e.\n'
        f'All members: {len(selected)}. Inside image footprint: {len(hull)}.\n'
        'The footprint is the convex hull of the observed multiple-image positions; selection uses galaxy centers.\n'
        'The orange sample is a subset of the blue sample. Histograms show raw counts in identical 0.05-wide bins, not probability densities.\n'
        f'Display limits are the all-member 0.5th and 99.5th percentiles: {display_limits[0]:.6f} to {display_limits[1]:.6f}. Full counts remain in the CSV; only the axis is cropped.\n'
        'Masses use the physical posterior-median parametric lens model, without the conditional GRF correction.\n')
    # All samples and normalizations share bins; no values are silently clipped.
    upper = max(3., np.ceil(data.ratio.max()*10)/10)
    bins = np.arange(0, upper+.10001, .1)
    rows = []
    with PdfPages(args.directory/'mass_ratio_histograms.pdf') as pdf:
        for name, selected in [('all',data),('image_hull',data[data.inside_image_hull])]:
            fig, axes = plt.subplots(2,3,figsize=(7.1,4.6),sharex=True,sharey=True)
            fig.subplots_adjust(wspace=0, hspace=0, left=.08, right=.99, bottom=.11, top=.99)
            maximum_count = 0
            for ax,(norm,group) in zip(axes.flat,selected.groupby('normalization_kpc')):
                r=group.ratio.to_numpy()
                q25,median,q75=np.quantile(r,[.25,.5,.75])
                product=(q75-q25)*abs(median-1)
                score=-np.log10(product) if product>0 else np.inf
                rows.append(dict(sample=name,normalization_kpc=norm,n=len(r),median=median,
                    p25=q25,p75=q75,iqr=q75-q25,absolute_median_bias=abs(median-1),Q=score))
                counts,_,_=ax.hist(r,bins=bins,color='tab:blue',alpha=.75,histtype='stepfilled',lw=.5)
                assert counts.sum()==len(r)
                maximum_count = max(maximum_count, counts.max())
                ax.axvline(1,color='.2',ls='--',lw=.8)
                ax.text(.96,.95,f'$R_{{e,20}}={norm:g}$ kpc\nMedian = {median:.3f}\n$p_{{25}},p_{{75}}={q25:.3f},{q75:.3f}$',
                    transform=ax.transAxes,ha='right',va='top',fontsize=8)
                ax.set_xlim(0,upper)
                ax.grid(False)
            axes[0,0].set_ylim(0, maximum_count*1.12)
            for ax in axes[-1]: ax.set_xlabel(r'$M/M_{\rm true}$')
            for ax in axes[:,0]: ax.set_ylabel('Number of galaxies')
            pdf.savefig(fig);plt.close(fig)
    table=pd.DataFrame(rows)
    table.to_csv(args.directory/'histogram_statistics.csv',index=False)
    provenance_path = args.directory/'provenance.json'
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        final = Path(provenance['run'])/'final'
        posterior_hash = hashlib.sha256((final/'artifacts/posterior.npz').read_bytes()).hexdigest()
        if posterior_hash != provenance['posterior_sha256']:
            raise ValueError('Aperture masses belong to a different posterior; refusing to export run macros')
        chosen = table[np.isclose(table.normalization_kpc,args.normalization)]
        payload = dict(normalization_kpc=args.normalization,
                       samples=chosen.set_index('sample').to_dict(orient='index'),
                       source_directory=str(args.directory.resolve()),
                       posterior_sha256=posterior_hash,
                       model=provenance['model'],grf=provenance['grf'],equation=provenance['equation'])
        (final/'tables/aperture_mass_histogram.json').write_text(json.dumps(payload,indent=2)+'\n')
        from lenscluster.results_tex import write_run_results
        print(write_run_results(final))
    (args.directory/'histogram_notes.md').write_text(
        '# Aperture-ratio histograms\n\nPage 1: all 340 saved population members. Page 2: 42 centers inside the image convex hull. '
        'Every panel uses identical 0.1-wide ratio bins. Dashed line: unity. No clipped histogram counts.\n\n'
        'Statistics follow the supplied stx2064.pdf, section 4.4 and equation 17: '
        'Q = -log10[(p75-p25)*abs(median-1)]. Larger Q is better. '
        'This score is singular at exactly unbiased medians; inspect width and tails too. '
        'The paper uses F814W<24 members in its comparison-map footprints, not our image convex hull; these are not identical samples.\n\n'
        'Six kpc is best by Q among the tested normalizations. Five kpc retains the smallest median absolute error inside the hull. '
        'These criteria differ. Neither identifies the true light sizes. The choice is tuned against truth and includes projected cluster background.\n')
    print(table.to_string(index=False))


if __name__=='__main__':
    main()
