"""Cache smoothed critical curves from saved physical posterior draws for ApJ cutouts."""
import hashlib
from pathlib import Path
from _covariance_plot_cli import load_run


def prepare(run, data):
    from lenscluster.fold_diagnostics import load_fold_data, save_fold_data, add_best_grf_critical_curves
    from lenscluster.grf_plotting import PLOT_FIELD_VERSION
    final = Path(run).resolve()/'final'
    cache = final/'tables/apj_fold_grf'
    digest = hashlib.sha256(PLOT_FIELD_VERSION.encode()+b'apj-physical-draws-quantile-curves-v3')
    for path in (final/'tables/fold_pairs_source_caustics.npz', final/'tables/fold_pairs_source_caustics.json',
                 final/'artifacts/posterior.npz'):
        digest.update(path.read_bytes())
    key = digest.hexdigest()
    if (cache/'fold_pairs_source_caustics.json').exists():
        saved = load_fold_data(cache)
        if saved.get('apj_grf_fingerprint') == key:
            return saved
    config, state, evaluator, posterior, paths = load_run(run, physical_draw_indices=data['draw_indices'])
    add_best_grf_critical_curves(state, evaluator, posterior, data,
                               system_block_fields=True, critical_only=True)
    if not all('critical_grf_best' in s for s in data['systems']):
        raise ValueError('No GRF critical curves available for this model')
    data['critical_overlay'] = 'Parametric and parametric + conditional-mean GRF; smooth potential derivatives; separate pointwise bands; system-specific fields when gp_blocks=system'
    data['apj_grf_fingerprint'] = key
    save_fold_data(data, cache)
    return data
