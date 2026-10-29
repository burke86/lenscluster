"""Publication macros from recorded results; no inference or plot generation."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

_DIGITS = ('Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine')
_ALIASES = {'ares': 'Ares', 'hera': 'Hera', 'm0416': 'MJZeroFourOneSix',
            'mj0416': 'MJZeroFourOneSix', 'macs0416': 'MJZeroFourOneSix',
            'm1206': 'MJOneTwoZeroSix', 'mj1206': 'MJOneTwoZeroSix',
            'as1063': 'ASOneZeroSixThree', 'mock': 'Mock'}


def command_prefix(name: str) -> str:
    key = re.sub(r'[^a-z0-9]', '', name.lower())
    if key in _ALIASES:
        return _ALIASES[key]
    words = re.findall(r'[A-Za-z]+|[0-9]', name)
    prefix = ''.join(_DIGITS[int(w)] if w.isdigit() else w[:1].upper() + w[1:] for w in words)
    if not prefix:
        raise ValueError('Results name must contain letters or digits')
    return prefix


def number(value, decimals=3) -> str:
    if value is None or not math.isfinite(float(value)):
        return r'\textnormal{not available}'
    return f'{float(value):.{decimals}f}'


def duration(seconds) -> str:
    if seconds is None or not math.isfinite(float(seconds)) or seconds < 0:
        return r'\textnormal{not available}'
    hours, remainder = divmod(int(math.floor(float(seconds) + .5)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return rf'\ensuremath{{{hours}\,\mathrm{{h}}\;{minutes}\,\mathrm{{min}}\;{seconds}\,\mathrm{{s}}}}'


def _read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def _write(path, prefix, metrics, comments):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['% Generated results. Do not edit; regenerate from saved artifacts.',
             '% Residuals are in arcseconds; fractions are percentages only when named Percent.',
             *['% ' + text for text in comments]]
    for suffix, value in metrics.items():
        command = prefix + suffix
        if not re.fullmatch('[A-Za-z]+', command):
            raise ValueError(f'Invalid TeX command: {command}')
        lines.append(rf'\newcommand{{\{command}}}{{{value}}}')
    path.write_text('\n'.join(lines) + '\n')
    return path


def write_run_results(final_dir: str | Path, *, name: str | None = None) -> Path:
    """Read final/{tables,artifacts} and export beside plots in final/figures.

    Missing quantities remain explicitly unavailable.
    """
    final = Path(final_dir)
    if not (final / 'artifacts/summary.json').is_file():
        raise FileNotFoundError(f'No saved run summary under {final / "artifacts"}')
    cfg = _read(final / 'artifacts/config.json')
    name = name or cfg.get('paths', {}).get('figure_prefix') or final.parent.name
    filename = re.sub(r'[^A-Za-z0-9_-]', '_', name)
    summary = _read(final / 'artifacts/summary.json')
    fit = _read(final / 'tables/run_summary.json').get('fit', {})
    diag = _read(final / 'artifacts/posterior_diagnostics.json') or summary.get('nuts', {})
    timing = _read(final / 'tables/runtime.json')
    metrics = {}
    for suffix, value in {
        'NImages': fit.get('observed_image_count', summary.get('images')),
        'NRecovered': fit.get('recovered_image_count'), 'NMissed': fit.get('missing_image_count'),
        'NFamilies': summary.get('families'), 'NFamiliesRecovered': fit.get('families_fully_recovered'),
        'NParameters': summary.get('n_parameters'), 'NMembers': summary.get('members'),
        'NExactMembers': summary.get('exact_members'), 'NDivergences': summary.get('divergences'),
        'NChains': cfg.get('runtime', {}).get('chains'),
        'NWarmupPerChain': cfg.get('sampling', {}).get('warmup'),
        'NSamplesPerChain': cfg.get('sampling', {}).get('samples'),
    }.items():
        metrics[suffix] = number(value, 0)
    members, exact = summary.get('members'), summary.get('exact_members')
    metrics['NFarFieldMembers'] = number(members-exact if members is not None and exact is not None else None, 0)
    metrics['RMS'] = number(fit.get('image_rms_arcsec'))
    metrics['MedianResidual'] = number(fit.get('image_median_residual_arcsec'))
    total, recovered = fit.get('observed_image_count'), fit.get('recovered_image_count')
    metrics['RecoveredPercent'] = number(100*recovered/total if total and recovered is not None else None, 1)
    metrics['MaxRhat'] = number(diag.get('max_r_hat'), 4)
    metrics['MinESS'] = number(diag.get('min_n_eff'), 1)
    for suffix, seconds in [('Runtime', timing.get('total')), ('SamplingRuntime', diag.get('elapsed_seconds'))]:
        metrics[suffix] = duration(seconds)
        metrics[suffix+'Seconds'] = number(seconds, 1)
        metrics[suffix+'Hours'] = number(seconds/3600 if seconds is not None else None, 3)
    extra = final / 'tables/image_recovery_extra_images.csv'
    metrics['NExtraImages'] = number(None, 0)
    if extra.exists():
        import pandas as pd
        try:
            metrics['NExtraImages'] = str(len(pd.read_csv(extra)))
        except pd.errors.EmptyDataError:
            metrics['NExtraImages'] = '0'
    return _write(final / 'figures' / f'{filename}_results.tex', command_prefix(name), metrics,
                  ['Best-fit image recovery; no reduced chi-square exported.',
                   'Runtime is recorded full-run wall time, including initialization and outputs.',
                   'SamplingRuntime is the saved sampler elapsed time, not total run time.',
                   'NSamplesPerChain is configured sampling iterations before thinning.',
                   'Unavailable values are not inferred from modification timestamps.'])


def write_mock_results(campaign, output: Path, *, rhat_cut: float, min_ess: float) -> Path:
    """Aggregate definitions match the residual figure, not a pooled image RMS."""
    import numpy as np
    import pandas as pd
    selected = campaign.summary[campaign.summary.seed.isin(campaign.seeds)]
    rms, medians = [], []
    for _, group in campaign.residual_draws.groupby(['seed', 'draw_index']):
        values = pd.to_numeric(group.image_residual_arcsec, errors='coerce').to_numpy(float)
        values = values[group.recovered.astype(bool).to_numpy() & np.isfinite(values) & (values >= 0)]
        if len(values):
            rms.append(np.sqrt(np.mean(values**2)))
            medians.append(np.median(values))
    metrics = {'NCompleted': str(len(campaign.summary)), 'NIncluded': str(len(campaign.seeds)),
               'NExcluded': str(len(campaign.summary)-len(campaign.seeds)),
               'RhatCut': number(rhat_cut, 2), 'MinESSCut': number(min_ess, 0),
               'RMS': number(np.median(rms) if rms else None),
               'MedianResidual': number(np.median(medians) if medians else None),
               'Runtime': duration(None), 'RuntimeSeconds': number(None),
               'RuntimeHours': number(None)}
    for suffix, col in [('NImages','observed_image_count'),('NRecovered','recovered_image_count'),('NMissed','missing_image_count')]:
        values = pd.to_numeric(selected[col], errors='coerce') if col in selected else None
        metrics[suffix] = number(values.sum() if values is not None and values.notna().all() else None, 0)
    return _write(output / 'mock_results.tex', 'Mock', metrics,
                  ['Counts sum best-fit images across included mocks.',
                   'RMS and MedianResidual are medians across recovered-image mock/draw groups.',
                   'Selection: max Rhat strictly below RhatCut; min ESS at least MinESSCut.',
                   'If the existing plotting fallback includes all mocks, NIncluded reflects that fallback.',
                   'No campaign runtime inferred: concurrent/resumed runs cannot be timed by summing draws.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('final_dirs', nargs='+', type=Path)
    args = parser.parse_args()
    for path in args.final_dirs:
        print(write_run_results(path))


if __name__ == '__main__':
    main()
