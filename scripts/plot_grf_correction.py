"""Export the GRF overview's middle panel from cached field maps only."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import numpy as np
    import pandas as pd
    from lenscluster.field_plots import render_grf_correction
    from lenscluster.member_catalog import load_member_catalog
    from lenscluster.sky import radec_to_offsets_arcsec
    from lenscluster.jwst_rgb_figure import marker_photometry
    from lenscluster.utils import log10_dpie_mass_msun
    final = args.run_directory.resolve() / 'final'
    source = final / 'tables/conditional_field_maps.npz'
    config = json.loads((final / 'artifacts/config.json').read_text())
    prefix = config['paths'].get('figure_prefix')
    output = final / 'figures' / f'{prefix + "_" if prefix else ""}grf_convergence.pdf'
    with np.load(source) as archive:
        data = {key: archive[key] for key in archive.files}
    members = pd.read_csv(final / 'tables/member_diagnostics.csv')
    data['member_xy'] = members[['x_centre','y_centre']].to_numpy()
    state = json.loads((args.run_directory / 'init/artifacts/state.json').read_text())
    with np.load(final / 'artifacts/posterior.npz') as archive:
        physical = archive['physical'][int(data['draw_index'])]
    fitted = dict(zip((s['name'] for s in state['parameter_specs']),physical))
    model = config['model']
    centres, labels = [], []
    for halo in model['large_halos']:
        hid = halo['id']
        if hid+'.v_disp' not in fitted:
            continue  # Exclude fixed gas components and external shear.
        centres.append([fitted.get(hid+'.'+axis,halo[axis]) for axis in ('x_centre','y_centre')])
        labels.append({'1':'H1','2':'H2','3':'H3','4':'H4','5':'FG',
                       'los736':'BG','dark_s':'Dark'}.get(hid,hid))
    ref = model['reference']
    reference = (ref['reference'],ref['ra0_deg'],ref['dec0_deg'])
    images = pd.read_csv(final / 'tables/image_fit_quality.csv')
    np.testing.assert_allclose(images[['x_obs_arcsec','y_obs_arcsec']].to_numpy(),data['xy'],atol=1e-9)
    photometry = marker_photometry(images,reference,
        Path(__file__).resolve().parents[1]/'src/lenscluster/data/m0416_image_f200w_photometry.csv')
    data['image_sizes'] = photometry.marker_area_pt2.to_numpy()
    with np.load(final / 'tables/member_diagnostic_draws.npz') as archive:
        deviations = archive['deviations'][int(data['draw_index'])]
    assert deviations.shape == (len(members),2)
    masses = []
    for j,row in members.iterrows():
        pop = row.potfile_id
        sigma = fitted[pop+'.sigma']*row.luminosity_ratio**fitted[pop+'.alpha_sigma']*np.exp(deviations[j,0])
        cut = fitted[pop+'.cutkpc']*row.luminosity_ratio**fitted[pop+'.beta_radius']*np.exp(deviations[j,1])
        masses.append(float(10**log10_dpie_mass_msun(sigma,cut)))
    masses = np.asarray(masses)
    assert np.isfinite(masses).all() and (masses>0).all()
    data['member_sizes'] = np.clip(10*np.sqrt(masses/1e11),3,70)
    marker_table = members[['member_id','x_centre','y_centre']].copy()
    marker_table['mass_msun'] = masses
    marker_table['marker_area_pt2'] = data['member_sizes']
    marker_table.to_csv(output.with_name(output.stem+'_member_markers.csv'),index=False)
    photometry.to_csv(output.with_name(output.stem+'_image_markers.csv'),index=False)
    populations = {p['id']:p for p in model['member_populations']}
    for halo in model.get('independent_member_halos',[]):
        population = populations[halo['population_id']]
        catalog = load_member_catalog(population['catalog_path'],reference,band=population.get('band'))
        row = catalog.loc[catalog.id.astype(str)==str(halo['catalog_id'])].iloc[0]
        x,y = radec_to_offsets_arcsec(np.array([row.ra]),np.array([row.dec]),reference[1],reference[2])
        hid = halo['id']
        centres.append([fitted.get(hid+'.x_centre',x[0]),fitted.get(hid+'.y_centre',y[0])])
        labels.append('BCG '+str(halo['catalog_id']))
    data['halo_xy'],data['halo_labels'] = np.asarray(centres),labels
    render_grf_correction(data, output)
    output.with_suffix('.json').write_text(json.dumps(dict(
        source=str(source),draw_index=int(data['draw_index']),
        member_positions=len(members),independent_halo_labels=labels,
        image_size='area = clip(16*10**(-0.2*(m_F200W-26)),9,64) pt^2; unavailable magnitudes use 9',
        image_measured_magnitudes=int(photometry.valid_photometry.sum()),
        member_size='area = clip(10*sqrt(M/1e11 Msun),3,70) pt^2; total dPIE mass at map draw',
        independent_halo_positions=centres,
        reference_redshift=float(data['reference_redshift']),
        panel='Middle panel of total_grf_dimensionless_power; unchanged cached convergence correction',
        colormap='vik',normalization='same symmetric zero-centered limits as the three-panel figure'),indent=2)+'\n')
    print(output)


if __name__ == '__main__':
    main()
