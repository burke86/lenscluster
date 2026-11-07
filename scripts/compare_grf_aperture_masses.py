"""Repeat the cached Hera size-normalization sweep with its conditional mean GRF.

No inference and no change to standard output settings. Reconstruct physical
median sources before conditioning; use the exact same parametric masses and
truth apertures as the original sweep, verified against the posterior hash.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from lenscluster import artifacts
from lenscluster.state import build_state, with_member_classes
from lenscluster.model import convert_theta_to_latent
from lenscluster.evaluator import Evaluator
from lenscluster.jax_cosmology import dpie_sigma0_factor_from_config
from lenscluster.aperture_mass_histogram import render_histogram, aperture_masses
from lenscluster.plot_style import PAPER_STYLE


def render_plots(out,parametric,truth,grf,norms,valid,inside,summary,table):
    plt.style.use(Path(__file__).resolve().parents[1]/'src/lenscluster/style.mplstyle');plt.rcParams.update(PAPER_STYLE)
    with PdfPages(out/'radius_histograms.pdf') as pdf:
        for label,select in [('all',valid),('image_hull',valid&inside)]:
            fig,axes=plt.subplots(3,3,figsize=(7.1,6.3),sharex=True,sharey=True)
            fig.subplots_adjust(left=.11,right=.99,bottom=.08,top=.98,wspace=0,hspace=0)
            limits=np.quantile(np.r_[(parametric/truth)[:,select].ravel(),((parametric+grf)/truth)[:,select].ravel()],[.005,.995])
            all_ratios=np.r_[(parametric/truth)[:,select].ravel(),((parametric+grf)/truth)[:,select].ravel()]
            bins=np.arange(np.floor(all_ratios.min()*20)/20,np.ceil(all_ratios.max()*20)/20+.05001,.05)
            for k,(norm,ax) in enumerate(zip(norms,axes.flat)):
                for mass,color,text in [(parametric,'tab:orange','Parametric'),(parametric+grf,'tab:blue','With GRF')]:
                    counts,_=np.histogram(mass[k,select]/truth[k,select],bins)
                    assert counts.sum()==int(select.sum())
                    ax.stairs(counts,bins,color=color,lw=1.2,label=text)
                ax.axvline(1,color='.3',ls='--',lw=.7)
                ax.text(.95,.95,f'{norm:g} kpc',transform=ax.transAxes,ha='right',va='top',fontsize=8, bbox=dict(facecolor='white',edgecolor='none',pad=1,alpha=.85))
                ax.set_xlim(limits);ax.grid(False)
            axes[0,0].legend(frameon=False,fontsize=6,loc='upper left')
            for ax in axes[-1]:ax.set_xlabel(r'$M/M_{\rm true}$')
            for ax in axes[:,0]:ax.set_ylabel('Number of galaxies')
            pdf.savefig(fig);plt.close(fig)
    selected=summary[(summary.model=='with_grf')&(summary['sample']=='all')]
    best=float(selected.sort_values('Q',ascending=False).iloc[0].normalization_kpc)
    render_histogram(table[table.normalization_kpc==best],out/'hera_all_vs_image_footprint_histogram.pdf')
    return best


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline',type=Path)
    parser.add_argument('--render-only', action='store_true')
    args=parser.parse_args()
    jax.config.update('jax_enable_x64',True)
    started=time.perf_counter()
    base=args.baseline;out=base/'grf_correction';out.mkdir(exist_ok=True)
    if args.render_only:
        cache=np.load(out/'aperture_arrays.npz')
        summary=pd.read_csv(out/'normalization_statistics.csv')
        table=pd.read_csv(out/'member_aperture_masses.csv')
        first=table[table.normalization_kpc==cache['normalizations_kpc'][0]]
        render_plots(out,cache['parametric_mass'],cache['truth_mass'],cache['grf_mass'],cache['normalizations_kpc'],
                     first.complete_aperture.to_numpy(),first.inside_image_hull.to_numpy(),summary,table)
        return
    provenance=json.loads((base/'provenance.json').read_text())
    run=Path(provenance['run']);init=run/'init/artifacts';final=run/'final'
    assert hashlib.sha256((final/'artifacts/posterior.npz').read_bytes()).hexdigest()==provenance['posterior_sha256']
    data=np.load(base/'aperture_arrays.npz')
    config=artifacts.load_config(final/'artifacts')
    classes=json.loads((init/artifacts.CLASSIFICATION_FILE).read_text())
    state=build_state(config,drop_member_ids=classes.get('dropped_members',()))
    centered=set(classes.get('centered_members',[]))
    state=with_member_classes(state,artifacts.load_exact_positions(init),classes,
        centered=np.array([m in centered for m in state.members.member_ids]))
    state.initial_theta=artifacts.load_start(init)
    saved=json.loads((init/artifacts.STATE_FILE).read_text())
    assert [s.name for s in state.parameter_specs]==[s['name'] for s in saved['parameter_specs']]
    assert list(state.members.member_ids)==saved['member_ids']==data['member_ids'].tolist()
    print('Building evaluator and reconstructing physical median sources',flush=True)
    evaluator=Evaluator(state,config.likelihood.likelihood_mode)._padded_evaluator()
    assert evaluator.gp_knee and state.gp_blocks=='all'
    posterior=artifacts.load_posterior(final/'artifacts')
    physical=np.median(posterior.physical,axis=0)
    np.testing.assert_allclose(physical,data['median_physical'],rtol=0,atol=0)
    theta=convert_theta_to_latent(physical,state.parameter_specs)
    ix=np.asarray(state.images.source_x_param_index);iy=np.asarray(state.images.source_y_param_index)
    theta[np.r_[ix,iy]]=0
    whitening=jax.jit(lambda t:evaluator.statistics(t)['whitening'])(theta)
    layout=evaluator.gp;mask=np.asarray(layout['knot_mask']);family=np.asarray(layout['knot_family'])
    source=np.stack((physical[ix],physical[iy]),-1)
    target=np.where(mask[...,None],source[np.where(mask,family,0)],0).reshape(mask.shape[0],-1)
    offsets=np.linalg.solve(np.asarray(whitening['cholesky']),
        (target-np.asarray(whitening['centroid']))[...,None])[...,0].reshape(*mask.shape,2)
    real=offsets[np.asarray(layout['real_block']),np.asarray(layout['real_knot'])]
    real_family=np.asarray(layout['real_family'])
    theta[ix[real_family]]=real[:,0];theta[iy[real_family]]=real[:,1]
    reconstructed=np.asarray(jax.jit(evaluator.reported_physical)(theta))
    np.testing.assert_allclose(reconstructed,physical,rtol=1e-10,atol=1e-10)
    correction=evaluator.gp_deflection_correction(theta)
    if correction is None:raise ValueError('Run does not support a conditional shared mass field')
    z=float(provenance['z_source'])
    ref=float(dpie_sigma0_factor_from_config(state.z_lens,state.gp_reference_redshift,state.cosmo_config))
    target_factor=float(dpie_sigma0_factor_from_config(state.z_lens,z,state.cosmo_config))
    weight=target_factor/ref/float(evaluator.gp_weights_per_family[0])
    members=pd.read_csv(final/'tables/member_diagnostics.csv',dtype={'member_id':str})
    assert members.member_id.tolist()==data['member_ids'].tolist()
    centers=members[['x_centre','y_centre']].to_numpy()
    radii=data['radii_arcsec'];norms=data['normalizations_kpc']
    conversion=provenance['mass_factor_msun_per_kappa_arcsec2']
    truth=data['truth_mass'];parametric=data['model_mass']
    valid=data['complete_aperture'].copy();inside=data['inside_image_hull']
    image_table=pd.read_csv(final/'tables/diagnostic_image_index.csv')
    image_xy=image_table[['x_obs_arcsec','y_obs_arcsec']].to_numpy()
    for norm in (7.,8.,10.):
        print(f'Extending exact parametric and truth masses to R20={norm:g} kpc',flush=True)
        extended=aperture_masses(state,evaluator,physical,members,image_xy,config.truth,normalization_kpc=norm)
        valid &= extended.usable.to_numpy()
        norms=np.r_[norms,norm]
        radii=np.concatenate([radii,extended.aperture_arcsec.to_numpy()[None]])
        truth=np.concatenate([truth,extended.truth_mass_msun.to_numpy()[None]])
        parametric=np.concatenate([parametric,extended.model_mass_msun.to_numpy()[None]])
    print(f'Common valid sample: {valid.sum()} members; {(valid&inside).sum()} in image hull',flush=True)
    def fields(points):
        result=[]
        for start in range(0,len(points),1024):
            block=points[start:start+1024];n=len(block)
            block=np.pad(block,((0,1024-n),(0,0)),mode='edge')
            values=correction(block[:,0],block[:,1],np.zeros(1024,dtype=int))
            result.append(np.stack(values,-1)[:n]*weight)
        return np.concatenate(result)
    def integrate(nt):
        angle=(np.arange(nt)+.5)*2*np.pi/nt
        n=np.stack((np.cos(angle),np.sin(angle)),-1)
        masses=[]
        for norm,r in zip(norms,radii):
            points=centers[:,None]+r[:,None,None]*n
            f=fields(points.reshape(-1,2)).reshape(len(centers),nt,6)
            masses.append(np.pi*r*np.mean(np.sum(f[:,:,:2]*n,axis=-1),axis=1)*conversion)
            print(f'GRF boundary: R20={norm:g} kpc, {nt} angular samples',flush=True)
        return np.array(masses)
    coarse=integrate(256);grf=integrate(512)
    error=np.abs(grf-coarse)/truth
    if np.nanmax(error[:,valid])>.001:
        old=grf;grf=integrate(1024);error=np.abs(grf-old)/truth
    if np.nanmax(error[:,valid])>.001:raise ValueError('GRF aperture quadrature did not converge to 0.1% of truth mass')
    # Independent area integration of trace(d alpha/d theta)/2 at representative apertures.
    ids=np.unique(np.r_[0,np.flatnonzero(inside)[::max(1,int(inside.sum())//3)],len(centers)//2])
    ids=ids[valid[ids]];checks=[]
    for g in ids:
        k=len(norms)-1;r=radii[k,g]
        estimates=[]
        for nr,nt in [(48,192),(96,384)]:
            q,w=np.polynomial.legendre.leggauss(nr);rad=(q+1)/2
            angle=(np.arange(nt)+.5)*2*np.pi/nt
            unit=rad[:,None,None]*np.stack((np.cos(angle),np.sin(angle)),-1)[None]
            f=fields((centers[g]+r*unit).reshape(-1,2))
            weights=np.broadcast_to((w*rad/2)[:,None],(nr,nt)).ravel()*2*np.pi/nt
            estimates.append(r*r*np.dot((f[:,2]+f[:,5])/2,weights)*conversion)
        checks.append(dict(member_id=str(data['member_ids'][g]),boundary_mass=float(grf[k,g]),
            area_mass=float(estimates[-1]),area_change_over_truth=float(abs(estimates[-1]-estimates[0])/truth[k,g]),
            boundary_area_difference_over_truth=float(abs(grf[k,g]-estimates[-1])/truth[k,g])))
    assert max(c['boundary_area_difference_over_truth'] for c in checks)<.001
    rows=[];histograms=[]
    for k,norm in enumerate(norms):
        for mode,mass in [('parametric',parametric),('with_grf',parametric+grf)]:
            ratio=mass[k]/truth[k]
            for label,select in [('all',valid),('image_hull',valid&inside)]:
                values=ratio[select];lo,med,hi=np.quantile(values,[.25,.5,.75])
                rows.append(dict(normalization_kpc=norm,model=mode,sample=label,n=int(select.sum()),
                    median=med,p25=lo,p75=hi,iqr=hi-lo,Q=-np.log10((hi-lo)*abs(med-1)),
                    median_absolute_error=np.median(abs(values-1))))
        histograms.append(pd.DataFrame(dict(member_id=data['member_ids'],normalization_kpc=norm,
            complete_aperture=valid,inside_image_hull=inside,ratio=(parametric[k]+grf[k])/truth[k],
            parametric_ratio=parametric[k]/truth[k],grf_mass_msun=grf[k],truth_mass_msun=truth[k],
            total_mass_msun=parametric[k]+grf[k],grf_integration_error_over_truth=error[k])))
    summary=pd.DataFrame(rows);table=pd.concat(histograms,ignore_index=True)
    summary.to_csv(out/'normalization_statistics.csv',index=False)
    table.to_csv(out/'member_aperture_masses.csv',index=False)
    np.savez_compressed(out/'aperture_arrays.npz',grf_mass=grf,parametric_mass=parametric,truth_mass=truth,
        normalizations_kpc=norms,member_ids=data['member_ids'],theta=theta,physical=physical)
    best=render_plots(out,parametric,truth,grf,norms,valid,inside,summary,table)
    metadata=dict(baseline=str(base.resolve()),posterior_sha256=provenance['posterior_sha256'],
        correction='conditional mean GRF at componentwise physical median, including physical median source positions; not posterior mean',
        source_roundtrip_max=float(np.max(abs(reconstructed-physical))),z_source=z,redshift_weight=weight,
        projected_modes=list(evaluator.gp_project_modes),boundary_error_over_truth_max=float(np.nanmax(error[:,valid])),
        independent_area_checks=checks,best_all_Q_normalization_kpc=best,elapsed_seconds=time.perf_counter()-started)
    (out/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(summary.to_string(index=False),flush=True)
    lines=['# Hera aperture masses with the conditional GRF','',metadata['correction'],'',
        'F160W=20 size normalizations: 1, 2, 3, 4, 5, 6 kpc from the baseline, extended to 7, 8 and 10 kpc. Aperture radius is twice Re.',
        'Best means highest paper Q = -log10(IQR * abs(median - 1)); tuned against truth, not a size measurement.',
        'No posterior field uncertainty is included. Parametric-only automatic outputs are unchanged.',
        'radius_histograms.pdf: page 1 all members; page 2 centers inside the image hull. Central 99% display.',
        f'hera_all_vs_image_footprint_histogram.pdf uses the best all-member Q normalization, {best:g} kpc.',
        '', '```',summary.to_string(index=False),'```','', '```json',json.dumps(metadata,indent=2),'```']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
