"""Plain figures and a self-contained report generated only when invoked."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .common import OUT,plot_style,read_json,write_json

def plots():
    status=[]
    def save(name,draw):
        try:
            with plot_style() as plt:
                figure=draw(plt);figure.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(figure)
            status.append(dict(plot=name,status='complete'))
        except Exception as error:status.append(dict(plot=name,status='unavailable',reason=repr(error)))
    if (OUT/'residuals.npz').exists():
        def residual(plt):
            data=np.load(OUT/'residuals.npz');fig,ax=plt.subplots(figsize=(4,3))
            for name,color in [('map','tab:blue'),('posterior_median','tab:orange')]:
                if name+'_residual' not in data:continue
                x=data[name+'_residual'];x=np.sort(x[np.isfinite(x)])
                ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',color=color,label=name.replace('_',' '))
            ax.axvline(.43,color='gray',ls=':',label='Published RMS (reference)')
            ax.set(xlabel='Image residual [arcsec]',ylabel='Recovered-image cumulative fraction');ax.legend(fontsize=8)
            return fig
        save('residuals',residual)
    if (OUT/'baseline_nuts.npz').exists():
        def corner(plt):
            import corner
            data=np.load(OUT/'baseline_nuts.npz');names=data['names'];keep=np.flatnonzero(~np.char.startswith(names,'source.'))
            x=data['reported'].reshape(-1,len(names))[:,keep]
            best=np.load(OUT/'baseline_map.npz')['reported'][keep]
            return corner.corner(x,labels=names[keep],truths=best,truth_color='tab:red',color='tab:blue',
                quantiles=[.16,.5,.84],show_titles=False,plot_datapoints=False,label_kwargs={'fontsize':8})
        save('posterior_corner',corner)
    if (OUT/'correlations.npz').exists():
        def correlations(plt):
            d=np.load(OUT/'correlations.npz');keep=np.flatnonzero(~np.char.startswith(d['names'],'source.'))
            fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
            for ax,key in zip(axes,['laplace','nuts']):
                artist=ax.imshow(d[key][np.ix_(keep,keep)],vmin=-1,vmax=1,cmap='RdBu_r')
                ax.set_xticks(np.arange(len(keep)),d['names'][keep],rotation=90,fontsize=5)
                ax.set_yticks(np.arange(len(keep)),d['names'][keep],fontsize=5)
                ax.text(.03,.97,key,transform=ax.transAxes,va='top',bbox=dict(facecolor='white',edgecolor='none'))
            fig.colorbar(artist,ax=axes,label='Correlation',shrink=.8);return fig
        save('correlations',correlations)
    if (OUT/'cosmography/f_summary.csv').exists():
        def ratios(plt):
            d=pd.read_csv(OUT/'cosmography/f_summary.csv',dtype={'family_id':str});fig,ax=plt.subplots(figsize=(5,3.4))
            free=~d.fixed.astype(bool);s=d[free]
            ax.errorbar(s.z,s['median'],yerr=[s['median']-s.p16,s.p84-s['median']],fmt='o',ms=3,lw=.5,label='Free family ratios')
            r=d[~free];ax.plot(r.z,r['median'],'*',color='tab:red',ms=8,label='Fixed reference')
            from .distances import flat_wcdm_lensing_efficiency
            z=np.linspace(.5,7,200);ax.plot(z,np.asarray(flat_wcdm_lensing_efficiency(.396,z,70.,.3,-1.)),color='black',lw=1,label='Fiducial geometry')
            ax.set(xlabel='Spectroscopic redshift (label only)',ylabel=r'$D_{ds}/D_s$',ylim=(0,1));ax.legend(fontsize=8);return fig
        save('distance_ratios',ratios)
    if (OUT/'cosmography/source_area.npz').exists():
        def area(plt):
            d=np.load(OUT/'cosmography/source_area.npz');fig,ax=plt.subplots(figsize=(4,3))
            ax.plot(d['z'],np.where(d['best_valid'],d['best'],np.nan),color='black',label='Optimized model')
            # Do not summarize a partially converged set as the full posterior.
            good=d['draw_valid'].all(0);q=np.full((3,len(d['z'])),np.nan)
            if good.any():q[:,good]=np.percentile(d['draws'][:,good],[16,50,84],axis=0)
            ax.plot(d['z'],q[1],color='tab:blue',label='Posterior median')
            ax.fill_between(d['z'],q[0],q[2],color='tab:blue',alpha=.2)
            ax.set(xlabel='Source redshift',ylabel=r'Area with at least two images [arcsec$^2$]');ax.legend(fontsize=8);return fig
        save('source_area',area)
    write_json(OUT/'plot_status.json',status);return status

def fmt(x):
    if x is None:return 'unavailable'
    if isinstance(x,str):return x.replace('|','/').replace('\n',' ')
    try:return f'{float(x):.4g}' if np.isfinite(float(x)) else 'unavailable'
    except (ValueError,TypeError):return str(x).replace('|','/').replace('\n',' ')

def table(columns,rows):
    return ['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']+[
        '| '+' | '.join(fmt(x) for x in row)+' |' for row in rows]

def package_arrays():
    arrays={}
    products={'baseline_nuts':'baseline_nuts.npz','laplace':'laplace.npz','map':'baseline_map.npz',
              'ratios':'cosmography/f_posteriors.npz','fisher':'cosmography/fisher.npz','area':'cosmography/source_area.npz',
              'correlations':'correlations.npz'}
    statuses={}
    for tag,relative in products.items():
        path=OUT/relative;statuses[tag]='available' if path.exists() else 'unavailable'
        if path.exists():
            with np.load(path,allow_pickle=False) as data:
                for key in data.files:arrays[tag+'__'+key]=data[key]
    arrays['product_status_json']=np.array(json.dumps(statuses))
    arrays['nuts_convergence_json']=np.array(json.dumps(read_json(OUT/'baseline_nuts.json',{})))
    np.savez_compressed(OUT/'results.npz',**arrays)

def description(path):
    name=path.name
    if name.endswith('.py'):return 'Unexecuted-at-authoring workflow source or validation check'
    if 'area_checkpoint' in name:return 'Resumable per-draw, per-redshift area calculations and resolution evidence'
    if path.parent.name=='failures':return 'Exception traceback from an unavailable step'
    if name=='results.npz':return 'Principal joint posterior, Laplace, ratio, Fisher and area arrays; prefixed keys'
    if name=='manifest.json':return 'Fingerprints, configuration, dependency versions and step status/timings'
    if name=='run.log':return 'Console transcript'
    if name.endswith('.pdf'):return 'Diagnostic figure; see filename and plot_status.json'
    if name.endswith('.csv'):return 'Numerical table with named columns and physical units defined in README'
    if name.endswith('.npz'):return 'Numerical checkpoint; named arrays loaded without pickle'
    if name.endswith('.json'):return 'Configuration, numerical diagnostics, timing or status metadata'
    if name.endswith('.cat'):return 'Separate selected-member catalog; original catalog remains untouched'
    return 'Workflow documentation or generated report'

def report():
    package_arrays();manifest=read_json(OUT/'manifest.json',{});audit=read_json(OUT/'catalog_audit.json',{})
    nuts=read_json(OUT/'baseline_nuts.json',{});comparison=read_json(OUT/'comparison.json',{})
    residual=read_json(OUT/'residuals.json',{});bench=read_json(OUT/'benchmarks.json',{})
    lines=['# MACS J0416 feasibility report','',
      'Scripts were written without execution or numerical validation. The statuses below describe this invocation, not the authoring session.',
      'This is a two-halo feasibility experiment, not a reproduction of the published four-halo Bergamini model. Published 0.43 arcsec RMS is a contextual reference, not a matched-model comparison.',
      'Members: brightest 30% union within 10 arcsec; exact dPIEs; fixed alpha=0.30, beta=0.60 and pivot=17.02, zero individual deviations. Sources have a fixed 100 arcsec physical Gaussian prior and likelihood-whitened coordinates. Astrometry=0.3 arcsec; source scatter is sampled.',
      'Physical parameter medians are computed jointly by columns, then evaluated as one model. They need not be a posterior mode. All inference times are synchronized; explicit kernel compilation is separate, sampler compilation is labeled where mixed.',
      '', '## Catalogs and selection',
      f"Images={audit.get('images','unavailable')}; families={audit.get('families','unavailable')}; members={audit.get('members','unavailable')}; retained={audit.get('selected_count','unavailable')}; luminosity fraction={fmt(audit.get('selected_luminosity_fraction'))}.",
      f"Redshift range={audit.get('redshift_range','unavailable')}; magnitude range={audit.get('magnitude_range','unavailable')}. Image columns: {audit.get('raw_image_columns',[])}. Member columns: {audit.get('raw_member_columns',[])}.",
      'Catalog flags: '+str(audit.get('failures',[])+audit.get('warnings',[])), '', '## Timings and status']
    lines+=table(['Step','Status','Wall s','Reason'],[(k,v.get('status'),v.get('wall_seconds'),v.get('error',v.get('reason',''))) for k,v in manifest.get('steps',{}).items()])
    lines+=['',f"NUTS: converged={nuts.get('converged','unavailable')}; maximum R-hat={fmt(nuts.get('max_rhat'))}; minimum bulk ESS={fmt(nuts.get('min_bulk_ess'))}; divergences={nuts.get('divergences','unavailable')}; seconds/minimum ESS including warmup={fmt(nuts.get('total_seconds_per_min_ess'))}; sampling seconds/median ESS={fmt(nuts.get('sampling_seconds_per_median_ess'))}.",
            'Timing details: '+json.dumps({k:v for k,v in nuts.items() if 'seconds' in k or 'compile_note'==k}),
            '', '## Exact image recovery']
    lines+=table(['Model','Recovered','Missed','Extra','RMS arcsec','Median arcsec','Common-subset RMS'],[
        (k,v.get('recovered'),v.get('missed'),v.get('extras'),v.get('rms'),v.get('median'),v.get('common_rms')) for k,v in residual.items() if isinstance(v,dict)])
    lines+=['Common subset size: '+str(residual.get('common_images','unavailable')),
            '', '## Parameter uncertainties',
            'Rows contain MAP; NUTS median [16%,84%]; NUTS standard deviation; Laplace standard deviation; Laplace/NUTS width ratio; R-hat; bulk/tail ESS. Source x and y share a row, in that order. Units: arcsec for source/center coordinates, kpc for radii, km/s for velocities, dimensionless for shape/shear. Laplace widths use valid transformed Gaussian draws; invalid fraction is reported below. Flags: U=Laplace width <0.7 NUTS; G=|skew|>0.5 or excess kurtosis>1.']
    parameter_path=OUT/'parameter_comparison.csv'
    if not parameter_path.exists():parameter_path=OUT/'baseline_nuts_diagnostics.csv'
    if parameter_path.exists():
        d=pd.read_csv(parameter_path);rows=[];used=set()
        def cell(r):
            return '; '.join([fmt(r.get('map')),f"{fmt(r.get('median'))} [{fmt(r.get('p16'))},{fmt(r.get('p84'))}]",fmt(r.get('std')),fmt(r.get('laplace_std')),
                             fmt(r.get('laplace_over_nuts')),fmt(r.get('rhat')),f"{fmt(r.get('ess_bulk'))}/{fmt(r.get('ess_tail'))}",
                             ('U' if r.get('laplace_under_30pct',False) else '')+('G' if r.get('non_gaussian',False) else '')])
        for _,r in d.iterrows():
            name=r.parameter
            if name in used:continue
            if name.startswith('source.'):
                partner=name[:-1]+'y';other=d[d.parameter==partner]
                if name.endswith('x') and len(other):
                    rows.append((name.rsplit('.',1)[0]+' (x / y)',cell(r)+' / '+cell(other.iloc[0])));used.add(partner)
                else:rows.append((name,cell(r)))
            else:rows.append((name,cell(r)))
        lines+=table(['Parameter','MAP; median [16,84]; sigma N; sigma L; L/N; Rhat; ESS bulk/tail; flags'],rows)
    else:lines+=['No NUTS parameter table is available.']
    lines+=['',f"Laplace leakage outside the empirical 95% NUTS Mahalanobis ellipsoid (unconstrained coordinates): {fmt(comparison.get('ellipsoid_outside_fraction'))}. This is not a non-Gaussian HPD region. Invalid Laplace fraction={fmt(comparison.get('invalid_laplace_fraction'))}. Per-parameter marginal 95% leakage is saved in parameter_comparison.csv.",
        'Full correlation matrices are in correlations.npz and the two correlation CSVs; the PDF displays global parameters only.',
        '', '## Performance projections',
        f"10,000 positions x 100 draws x 8 redshifts: synchronized field time={fmt(bench.get('field_execution_seconds'))} s; compile={fmt(bench.get('field_compile_seconds'))} s; pair throughput={fmt(bench.get('pair_comparisons_per_second'))} comparisons/s."]
    lines+=table(['Objects','Unordered pairs','Field s','Exhaustive pairs s','Total s'],[(n,v['unordered_pairs'],v['field_seconds'],v['exhaustive_pair_seconds'],v['total_seconds']) for n,v in bench.get('projections',{}).items()])
    lines+=['300-image refit projection: '+json.dumps(bench.get('refit_300',{})),
            'Fields scale linearly in objects, exhaustive pair screening quadratically. Screening projections repeat every pair for 100 draws and eight trial redshifts; they exclude candidate-list output and catalog-specific filtering.',
            '', '## Cosmography',
            'Free-ratio fit fixes family 0141 at the fiducial ratio at z=3.221. Other ratios have independent Uniform(0,1) priors; spectroscopic redshifts are labels only. Geometry forecasts vary f(z), hold the angular mass chart and H0 fixed, and contain no cosmological priors.',
            'Gaussian Fisher includes local image mean and covariance derivatives. Observed curvature is the Hessian of the full physical-source smoothed log likelihood, without source-coordinate Jacobians or priors. All source nuisances are marginalized. Null cosmology directions produce unavailable uncertainties.']
    forecasts=read_json(OUT/'cosmography/fisher.json',[])
    lines+=table(['Families','Sheet','Definition','sigma Om','sigma w','Status'],[(r.get('selection'),r.get('free_sheet'),r.get('kind'),r.get('sigma_omega_m'),r.get('sigma_w'),'valid' if r.get('valid') else r.get('reason')) for r in forecasts])
    lines+=['Cosmology-normalization correlations: '+json.dumps([{k:r[k] for k in ('selection','free_sheet','kind','normalization_cosmology_correlations') if k in r} for r in forecasts if r.get('valid')]),
            'Free-ratio NUTS status: '+json.dumps(read_json(OUT/'ratios_nuts.json',{})),
            '', 'Family ratios (reference has no sampling uncertainty):']
    fpath=OUT/'cosmography/f_summary.csv'
    if fpath.exists():
        f=pd.read_csv(fpath,dtype={'family_id':str})
        lines+=table(['Family','z','Best f','Median [16,84]','Std','Rhat','Bulk ESS'],[(r.family_id,r.z,r.best,f'{fmt(r["median"])} [{fmt(r.p16)},{fmt(r.p84)}]',r['std'],r.rhat,r.ess_bulk) for _,r in f.iterrows()])
    else:lines+=['Free-ratio posterior unavailable.']
    if (OUT/'cosmography/source_area.npz').exists():
        a=np.load(OUT/'cosmography/source_area.npz')
        lines+=['', 'Area [arcsec^2] with two or more distinct images inside the observed-image hull. Unconverged cells remain unavailable.']
        rows=[]
        for j,z in enumerate(a['z']):
            good=a['draw_valid'][:,j];q=np.percentile(a['draws'][:,j],[16,50,84]) if good.all() else [np.nan]*3
            rows.append((z,a['best'][j] if a['best_valid'][j] else None,f'{fmt(q[1])} [{fmt(q[0])},{fmt(q[2])}]',int(good.sum())))
        lines+=table(['z','Best','Posterior median [16,84]','Resolved draws'],rows)
    if not nuts.get('converged',False):
        recommendation='NUTS has not passed the recorded convergence criteria, so this run cannot establish that Laplace uncertainties are adequate. Improve identification and convergence before using either uncertainty estimate for decisions. Optimization-only refits can still be timed, but they do not validate uncertainty.'
    elif not comparison:
        recommendation='NUTS passed the recorded checks, but a valid Laplace comparison is unavailable. Use NUTS uncertainties until the joint curvature and transformed Gaussian approximation are established.'
    elif comparison.get('underestimated_parameters') or comparison.get('non_gaussian_parameters'):
        recommendation='Use optimization for rapid updates only when point predictions are sufficient. The recorded width or shape discrepancies argue for NUTS when uncertainties, cosmography, or ambiguous image associations matter. Laplace may initialize refits but should not replace the measured posterior.'
    else:
        recommendation='For this reduced model, the recorded convergence and width checks support Laplace as a provisional per-round approximation near the same mode. Use NUTS after changes to image associations, halo structure, or priors, and for final cosmography. This comparison does not establish repeated-data coverage or absence of other modes.'
    lines+=['','## Recommendation',recommendation,'','## Output inventory']
    # Include the report and inventory themselves, even on their first creation.
    paths=set(p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'archives' not in p.relative_to(OUT).parts and p.suffix!='.tmp')
    paths.update(OUT/x for x in ['REPORT.md','summary.md','inventory.md','results.npz'])
    inventory=[f'- `{p.relative_to(OUT)}`: {description(p)}.' for p in sorted(paths)]
    lines+=inventory
    (OUT/'inventory.md').write_text('# File inventory\n\n'+'\n'.join(inventory)+'\n')
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (OUT/'summary.md').write_text('# Feasibility summary\n\n'+lines[9]+'\n\n'+
        'Exact recovery: '+json.dumps(residual)+'\n\n'+recommendation+'\n\nSee REPORT.md for parameter, timing, ratio and forecast tables.\n')
    return dict(report_lines=len(lines),under_400_lines=len(lines)<400)
