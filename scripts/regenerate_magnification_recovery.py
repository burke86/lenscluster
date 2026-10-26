"""Regenerate only magnification recovery from saved final maps and image tables."""
from pathlib import Path
import argparse
import json
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directories', nargs='+', type=Path, help='Run directories containing final/')
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import pandas as pd
    from astropy.io import fits
    from astropy.wcs import WCS
    from lenscluster.exact_images import astropy_cosmology
    from lenscluster.truth_recovery import grid_arcsec
    from lenscluster.magnification_recovery_plot import write_magnification_recovery

    for run in args.run_directories:
        final = run / 'final'
        cfg = json.loads((final / 'artifacts/config.json').read_text())
        reference = cfg['model']['reference']
        reference = (3, reference['ra0_deg'], reference['dec0_deg'])
        cosmo = cfg['model']['cosmology']
        cosmo['class'] = cosmo.get('class_name', 'FlatLambdaCDM')
        provenance = json.loads((final / 'tables/image_catalog_mass_provenance.json').read_text())
        t, header = fits.getdata(final / 'fits/truth_recovery_mu_true.fits', header=True)
        m, mh = fits.getdata(final / 'fits/truth_recovery_mu_model_median.fits', header=True)
        import numpy as np
        if t.shape != m.shape or header['ZSOURCE'] != mh['ZSOURCE']:
            raise ValueError('Saved truth and model grids differ')
        state = SimpleNamespace(reference=reference)
        x, y = grid_arcsec(state, WCS(header), t.shape)
        mx, my = grid_arcsec(state, WCS(mh), m.shape)
        np.testing.assert_allclose(x, mx, rtol=0, atol=1e-8)
        np.testing.assert_allclose(y, my, rtol=0, atol=1e-8)
        truth = dict(x=x, y=y, truth={'mu': t}, model={'mu': {'median': m}}, z_source=header['ZSOURCE'])
        images = pd.read_csv(final / 'tables/image_fit_quality.csv', dtype={'image_label': str, 'family_id': str})
        write_magnification_recovery(truth, images, reference, provenance['z_lens'], astropy_cosmology(cosmo),
                                     SimpleNamespace(**cfg['truth']), final / 'figures', final / 'tables', cfg['paths']['figure_prefix'])


if __name__ == '__main__':
    main()
