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


def interval(median, lower_error, upper_error, decimals=1):
    """TeX numerical interval, to be used in math mode."""
    return (number(median, decimals) + '_{-' + number(lower_error, decimals)
            + '}^{+' + number(upper_error, decimals) + '}')


def m0416_table_metrics(final):
    """Saved-draw model comparison; R25 published Tables 4--5, A&A 696, A15 (2025)."""
    import numpy as np
    import pandas as pd
    summary_path = final / 'tables/posterior_summary.csv'
    state_path = final.parent / 'init/artifacts/state.json'
    if not summary_path.exists() or not state_path.exists():
        return {}
    table = pd.read_csv(summary_path).set_index('parameter')
    state = _read(state_path)
    scale = state['kpc_per_arcsec']
    result = {}
    components = [('HaloOne','1'), ('HaloTwo','2'), ('HaloThree','3'), ('HaloFour','4'),
                  ('Foreground','5'), ('BCGNorth','bcg_2526'), ('BCGSouth','bcg_3158'),
                  ('SouthernGalaxy','los736'), ('SouthernDark','dark_s')]
    for label, component in components:
        for suffix, field, factor, decimals in [('Sigma','v_disp',1,0),
                ('Core','core_radius_kpc',scale,1), ('Cut','cut_radius_kpc',scale,1)]:
            key = component+'.'+field
            if key in table.index:
                row = table.loc[key]
                result[label+suffix] = interval(row['median']/factor,
                    (row['median']-row.p16)/factor, (row.p84-row['median'])/factor, decimals)
    for label, field, decimals in [('ScalingAlpha','alpha_sigma',3), ('ScalingBeta','beta_radius',3),
            ('ScalingSigmaScatter','velocity_log_scatter',3), ('ScalingRadiusScatter','radius_log_scatter',2)]:
        key = 'potfile_1.'+field
        if key in table.index:
            row = table.loc[key]
            result[label] = interval(row['median'],row['median']-row.p16,row.p84-row['median'],decimals)
    result['ComparisonMagnitude'] = '17.02'
    posterior_path = final / 'artifacts/posterior.npz'
    members_path = final / 'tables/member_diagnostics.csv'
    if posterior_path.exists() and members_path.exists():
        with np.load(posterior_path) as archive:
            physical = archive['physical']
        index = {spec['name']:i for i,spec in enumerate(state['parameter_specs'])}
        for key in table.index:
            if key in index:
                np.testing.assert_allclose(np.median(physical[:,index[key]]), table.loc[key,'median'],rtol=1e-9,atol=1e-10)
        def v(field):
            return physical[:,index['potfile_1.'+field]]
        def quantile(key, values, decimals):
            low, median, high = np.quantile(values,[.16,.5,.84])
            result[key] = interval(median,median-low,high-median,decimals)
        members = pd.read_csv(members_path)
        group = members[members.potfile_id == 'potfile_1']
        pivots = group.catalog_mag + 2.5*np.log10(group.luminosity_ratio)
        pivot = float(pivots.median())
        np.testing.assert_allclose(pivots,pivot,atol=1e-8)
        quantile('ScalingSigma',v('sigma')*10**(.4*v('alpha_sigma')*(pivot-17.02)),0)
        quantile('ScalingCut',v('cutkpc')/scale*10**(.4*v('beta_radius')*(pivot-17.02)),1)
        deviations_path = final / 'tables/member_diagnostic_draws.npz'
        match = np.flatnonzero(members.member_id.to_numpy() == 'potfile_1:8971')
        if len(match)==1 and deviations_path.exists():
            j = int(match[0])
            with np.load(deviations_path) as archive:
                deviations = archive['deviations']
            if deviations.shape != (len(physical),len(members),2):
                raise ValueError('Member deviation draws do not align with saved posterior')
            # Saved deviations already include centering/scatter transformations.
            row = members.iloc[j]
            for axis, prefix in [(0,'velocity'),(1,'radius')]:
                np.testing.assert_allclose(np.quantile(deviations[:,j,axis],[.16,.5,.84]),
                    [row[prefix+'_p16'],row[prefix+'_median'],row[prefix+'_p84']],atol=1e-9)
            lum = float(row.luminosity_ratio)
            quantile('GalEightNineSevenOneSigma',v('sigma')*lum**v('alpha_sigma')*np.exp(deviations[:,j,0]),0)
            quantile('GalEightNineSevenOneCut',v('cutkpc')/scale*lum**v('beta_radius')*np.exp(deviations[:,j,1]),1)
    return result


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
    # Keep the median individual residual distinct from the median of draw RMSs.
    import numpy as np
    import pandas as pd
    parameter_path = final / 'tables/posterior_summary.csv'
    parameters = pd.read_csv(parameter_path).set_index('parameter') if parameter_path.exists() else pd.DataFrame()
    for suffix, parameter, decimals in [
        ('GPSigmaShared', 'images.sigma_shared', 3),
        ('GPSigmaWhite', 'images.sigma_white', 3),
        ('GPCorrelationLength', 'images.gp_ell', 1),
        ('GRFDeflectionAmplitude', 'images.gp_sigma_alpha', 2),
        ('GRFTurnoverScale', 'images.gp_knee', 1),
    ]:
        value = parameters.loc[parameter, 'median'] if parameter in parameters.index else None
        metrics[suffix] = number(value, decimals)
        metrics[suffix+'Interval'] = number(None)
        if parameter in parameters.index and {'p16', 'p84'}.issubset(parameters.columns):
            row = parameters.loc[parameter]
            metrics[suffix+'Interval'] = interval(value, value-row.p16, row.p84-value, decimals)
    metrics['GRFReferenceRedshift'] = number(cfg.get('likelihood', {}).get('gp_reference_redshift'), 1)
    metrics['PosteriorMedianRMS'] = number(None)
    metrics['NPosteriorRMSDraws'] = number(None, 0)
    rms_path = final / 'tables/posterior_image_rms.csv'
    if rms_path.exists():
        rms = pd.read_csv(rms_path)['image_rms_arcsec'].to_numpy(float)
        rms = rms[np.isfinite(rms)]
        metrics['PosteriorMedianRMS'] = number(np.median(rms) if len(rms) else None)
        metrics['NPosteriorRMSDraws'] = str(len(rms))
    quality_path = final / 'tables/image_fit_quality.csv'
    quality = pd.read_csv(quality_path) if quality_path.exists() else pd.DataFrame()
    for axis in ('x', 'y'):
        suffix = axis.upper()
        metrics['PositionCoverage'+suffix+'Percent'] = number(None, 0)
        metrics['NPositionCoverage'+suffix] = number(None, 0)
        columns = [axis+'_obs_arcsec', axis+'_model_q16', axis+'_model_q84']
        if all(column in quality for column in columns):
            values = quality[columns].to_numpy(float)
            values = values[np.isfinite(values).all(axis=1)]
            metrics['NPositionCoverage'+suffix] = str(len(values))
            if len(values):
                obs, low, high = values.T
                metrics['PositionCoverage'+suffix+'Percent'] = number(
                    100*np.mean((low <= obs) & (obs <= high)), 0)
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
    if 'm0416' in name.lower() or 'mj0416' in name.lower():
        metrics.update(m0416_table_metrics(final))
    aperture = _read(final / 'tables/aperture_mass_histogram.json')
    if aperture:
        metrics['ApertureEffectiveRadiusNormalizationKpc'] = number(aperture['normalization_kpc'], 1)
        for sample, label in [('all', 'All'), ('image_hull', 'ImageFootprint')]:
            row = aperture['samples'][sample]
            stem = 'ApertureMassRatio' + label
            metrics[stem + 'NMembers'] = str(int(row['n']))
            for field, suffix in [('median','Median'), ('p25','LowerQuartile'),
                                  ('p75','UpperQuartile'), ('iqr','IQR'), ('Q','Score')]:
                metrics[stem + suffix] = number(row[field], 3)
    return _write(final / 'figures' / f'{filename}_results.tex', command_prefix(name), metrics,
                  ['Best-fit image recovery; no reduced chi-square exported.',
                   'RMS and MedianResidual describe the best-fit recovered images.',
                   'GP and GRF parameter macros are posterior medians; angular scales are in arcseconds.',
                   'ApertureMassRatio macros, when present: galaxy-distribution median and quartiles, not posterior intervals.',
                   'Apertures use twice the estimated effective radius; normalization is Re at F160W=20 in kpc.',
                   'ImageFootprint means galaxy centers inside the observed-image convex hull. Score is -log10(IQR*abs(median-1)).',
                   'PosteriorMedianRMS is the median of finite saved per-draw RMSs, not MedianResidual.',
                   'PositionCoverageX/YPercent: observed coordinates inside saved pointwise 16-84% model intervals.',
                   'Coverage denominators NPositionCoverageX/Y include only finite observed coordinates and bounds.',
                   'These are in-sample diagnostics of the saved recovery model, not held-out predictive errors.',
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
                   'Convergence diagnostics: max Rhat strictly below RhatCut; min ESS at least MinESSCut.',
                   'Aggregates include all completed mocks by default; optional converged-only selection is recorded in campaign_selection.json.',
                   'No campaign runtime inferred: concurrent/resumed runs cannot be timed by summing draws.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('final_dirs', nargs='+', type=Path)
    args = parser.parse_args()
    for path in args.final_dirs:
        print(write_run_results(path))


if __name__ == '__main__':
    main()
