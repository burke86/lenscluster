"""Render the four-panel marginal covariance using saved best-fit diagnostics only."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import make_covariance_revision as m

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run_directory',type=Path);p.add_argument('--image',default='K1A.a');args=p.parse_args()
    run=args.run_directory.resolve();final=run/'final';tables=final/'tables'
    cfg=json.loads((final/'artifacts/config.json').read_text());lk=cfg['likelihood']
    specs=json.loads((run/'init/artifacts/state.json').read_text())['parameter_specs']
    with np.load(final/'artifacts/posterior.npz',allow_pickle=True) as d:
        best=int(np.nanargmax(d['log_prob']));physical=d['physical'][best]
    values={s['name']:float(v) for s,v in zip(specs,physical)}
    audit=json.loads((tables/'covariance_errors_both_planes.json').read_text());assert audit['best_draw']==best
    widths=pd.read_csv(tables/'covariance_errors_both_planes.csv');widths=widths[widths.image_label==args.image]
    direction=pd.read_csv(tables/'image_catalog_arc_directions_jwst.csv').set_index('image_label').loc[args.image]
    image=pd.read_csv(tables/'image_fit_quality.csv').set_index('image_label').loc[args.image]
    assert bool(direction.direction_defined)
    src=widths[(widths.plane=='source')&(widths.direction=='arc')].iloc[0]
    floor=float(lk['gp_floor_arcsec']);effective=float(src.model_arcsec)
    alpha=values['images.gp_sigma_alpha'];sa=np.sqrt(effective**2-floor**2);w=sa/alpha
    sm=np.hypot(float(image.sigma_inst_arcsec),values['images.sigma_white']);sh=values['images.sigma_shared']
    lt=float(direction.s_arc);lr=float(direction.s_perp);angle=float(np.degrees(np.arctan2(direction.e_arc_y,direction.e_arc_x)))
    errors=[]
    for _,row in widths.iterrows():
        l=float(row.singular_value_best);total=np.sqrt((sm**2+sh**2)*l*l+effective**2)
        if row.plane=='image':total/=l
        errors.append(abs(total-row.total_arcsec));np.testing.assert_allclose(total,row.total_arcsec,rtol=1e-11,atol=1e-12)
    m.OUT.mkdir(parents=True,exist_ok=True);m.plt.style.use(m.ROOT/'src/lenscluster/style.mplstyle');m.plt.rcParams.update(m.PAPER_STYLE)
    fig=m.mapping(sm=sm,sa=sa,sh=sh,lt=lt,lr=lr,angle_deg=angle,floor=floor)
    fig.savefig(m.OUT/'covariance_local_mapping.pdf');m.plt.close(fig)
    provenance=dict(run=str(run),draw=best,selection='maximum saved log_prob',image=args.image,z_source=float(image.z_source),
        sigma_inst=float(image.sigma_inst_arcsec),sigma_white=values['images.sigma_white'],sigma_meas=sm,sigma_shared=sh,
        sigma_alpha_reference=alpha,redshift_weight_inferred_from_saved_diagonal=w,sigma_alpha_at_image=sa,source_floor=floor,
        deflection_plus_floor=effective,lambda_t_abs=lt,lambda_r_abs=lr,angle_deg=angle,
        gp_knee=values['images.gp_knee'],gp_ell=values['images.gp_ell'],max_saved_total_width_error=max(errors),
        posterior_sha256=hashlib.sha256((final/'artifacts/posterior.npz').read_bytes()).hexdigest())
    (m.OUT/'latest_run_provenance.json').write_text(json.dumps(provenance,indent=2));print(json.dumps(provenance,indent=2))
    (m.OUT/'latest_run_caption.txt').write_text(
        r'\textbf{Image- and source-plane positional uncertainties.} Panels (a) and (b) show the marginal covariance ellipses for image '+args.image+
        f' at z={image.z_source:.4f}, using saved best-fit draw {best}. Ellipse semi-axes represent one standard deviation, not 68 percent joint-probability contours. '
        'Measurement includes the instrumental and independent white scatter. The orange curve combines the redshift-scaled deflection GRF and the independent source-plane floor in quadrature. '
        'Panels (c) and (d) vary the absolute tangential eigenvalue while holding the fitted amplitudes and source redshift fixed; circles mark the illustrated image. '
        'Panel (d) uses logarithmic axes and excludes zero, where the inverse-Jacobian mapping is undefined. These are marginal model uncertainties, not post-correction residuals or posterior uncertainty bands. '
        'The ellipse panels use different scales. Cross-image correlations are not displayed.\n')
if __name__=='__main__':main()
