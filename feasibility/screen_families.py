#!/usr/bin/env python
"""Fast held-out-family validation and calibrated pair screening. UNTESTED."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('JAX_ENABLE_X64','true')
os.environ.setdefault('MPLCONFIGDIR','/tmp/lenscluster-mpl')
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/lenscluster-numba')

def signature(settings):
    paths=[ROOT/'feasibility/baseline_map.npz',ROOT/'feasibility/members_selected.cat',
           ROOT/'data/Bergamini/M0416_Bergamini22/obs_arcs.cat',
           *sorted((ROOT/'feasibility').glob('*.py')),*sorted((ROOT/'src/lenscluster').rglob('*.py'))]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return hashlib.sha256(json.dumps(dict(settings=settings,files=hashes),sort_keys=True).encode()).hexdigest()

def load_known():
    import numpy as np
    from lenscluster.inputs import load_multiple_images_catalog,radec_to_offsets_arcsec
    from feasibility.catalogs import IMAGES,REFERENCE
    data=load_multiple_images_catalog(IMAGES,REFERENCE,sigma_inst_arcsec=.3)
    data['id']=data.image_label.astype(str);data['family_id']=data.family_id.astype(str)
    data['x'],data['y']=radec_to_offsets_arcsec(data.ra.to_numpy(),data.dec.to_numpy(),*REFERENCE[1:])
    if data.id.duplicated().any() or not np.isfinite(data[['x','y','catalog_z']].to_numpy()).all():raise ValueError('Malformed known catalog')
    if (data.groupby('family_id').catalog_z.nunique()!=1).any():raise ValueError('Inconsistent family redshifts')
    return data

def make_folds(data,nfolds,seed):
    import numpy as np
    groups=data.catalog_z.map(lambda z:f'{z:.4f}')
    anchor=set(groups[data.family_id=='0141'])
    if len(anchor)!=1:raise ValueError('Reference family 0141 must exist at one redshift')
    keys=[k for k in groups.unique() if k not in anchor]
    rng=np.random.default_rng(seed);rng.shuffle(keys)
    keys.sort(key=lambda k:-int((groups==k).sum()))
    load=np.zeros(nfolds,int);assignment={}
    for k in keys:
        fold=int(np.argmin(load));assignment[k]=fold;load[fold]+=int((groups==k).sum())
    folds=groups.map(lambda k:assignment.get(k,-1)).to_numpy(int)
    if np.any(load==0):raise ValueError('Not enough independent redshift groups for requested folds')
    return folds

def training_catalog(data,mask,path):
    from feasibility.catalogs import IMAGES
    labels=set(data.loc[mask,'id']);lines=IMAGES.read_text().splitlines()
    output=[line for line in lines if line.lstrip().startswith('#') or
            (line.strip() and line.split()[0] in labels)]
    path.write_text('\n'.join(output)+'\n')

def quick_fit(catalog,maxiter,output):
    import numpy as np
    import jax.numpy as jnp
    from scipy.optimize import minimize
    from feasibility.model import Model
    from feasibility.common import compile_call,write_json,write_npz
    from feasibility.runtime import progress
    model=Model(image_catalog=catalog);start=np.array(model.u0,copy=True)
    executable,_,timing=compile_call(model.vg,jnp.asarray(start))
    def objective(u):
        v,g=executable(jnp.asarray(u));return float(v),np.asarray(g)
    begin=time.perf_counter()
    with progress(total=maxiter,desc='Held-out MAP',unit='iter',position=1,leave=False) as bar:
        result=minimize(objective,start,jac=True,method='L-BFGS-B',callback=lambda x:bar.update(),
            options={'maxiter':maxiter,'ftol':1e-9,'gtol':1e-3,'maxls':30,'maxcor':20})
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():raise ValueError('No finite training fit')
    physical=np.asarray(model.ev.physical(model.full(jnp.asarray(result.x))))
    info=dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit),
              gradient_max=float(np.max(abs(result.jac))),potential=float(result.fun),
              optimizer_seconds=time.perf_counter()-begin,**timing,
              initialization='Configured starting model, never full-catalog fitted parameters')
    write_npz(output/'fit.npz',u=result.x,physical=physical,names=np.array(model.names))
    write_json(output/'fit.json',info)
    return model,physical,info

def score_subset(data,rows,beta,cov,minimum,efficiencies):
    import numpy as np
    import pandas as pd
    from feasibility.family_probability import eligible_pairs,pair_scores
    sub=data.iloc[rows].reset_index(drop=True);pairs=eligible_pairs(sub[['x','y']].to_numpy(),minimum)
    if not len(pairs):raise ValueError('No eligible pairs in held-out set')
    score,q,best=pair_scores(beta[:,rows],cov[:,rows],pairs)
    i,j=pairs.T
    return pd.DataFrame(dict(id_a=sub.id.to_numpy()[i],id_b=sub.id.to_numpy()[j],
        family_a=sub.family_id.to_numpy()[i],family_b=sub.family_id.to_numpy()[j],
        same_family=sub.family_id.to_numpy()[i]==sub.family_id.to_numpy()[j],
        log_overlap=score,min_mahalanobis2=q,best_efficiency=efficiencies[best]))

def validate(args,settings):
    import numpy as np
    import pandas as pd
    from feasibility.common import write_json,read_json
    from feasibility.distances import flat_wcdm_lensing_efficiency
    from feasibility.family_probability import geometry,fit_calibrator,probabilities,metrics
    from feasibility.runtime import progress
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    key=signature(settings);manifest=read_json(output/'validation_manifest.json')
    if manifest and (not args.resume or manifest['signature']!=key):
        raise ValueError('Existing validation: use --resume for unchanged inputs, or a new --output directory')
    write_json(output/'validation_manifest.json',dict(signature=key,settings=settings))
    data=load_known();folds=make_folds(data,args.folds,args.seed)
    data.assign(fold=folds).to_csv(output/'fold_assignment.csv',index=False)
    f_lo,f_hi=np.asarray(flat_wcdm_lensing_efficiency(.396,np.array([args.z_min,args.z_max]),70.,.3,-1.))
    efficiencies=np.linspace(f_lo,f_hi,args.z_grid);tests=[];calibrations=[];status=[]
    for k in progress(range(args.folds),desc='Held-out folds',unit='fold'):
        directory=output/f'fold_{k}';directory.mkdir(exist_ok=True)
        cached=read_json(directory/'status.json')
        if args.resume and cached and cached.get('complete'):
            test=pd.read_csv(directory/'test_pairs.csv',dtype={'id_a':str,'id_b':str,'family_a':str,'family_b':str})
            cal=pd.read_csv(directory/'calibration_pairs.csv',dtype={'id_a':str,'id_b':str,'family_a':str,'family_b':str})
            tests.append(test);calibrations.append(cal);status.append(cached);continue
        begin=time.perf_counter()
        try:
            calibration_fold=(k+1)%args.folds
            training=(folds!=k)&(folds!=calibration_fold)
            path=directory/'training_images.cat';training_catalog(data,training,path)
            model,physical,fit=quick_fit(path,args.maxiter,directory)
            # No validation/test family, position or redshift enters this fit.
            # Held-out spec-z enters fold grouping only, never geometric scores.
            beta,cov,field_time=geometry(model,physical,data[['x','y']].to_numpy(),efficiencies,args.sigma_obs)
            cal=score_subset(data,np.flatnonzero(folds==calibration_fold),beta,cov,args.min_separation,efficiencies)
            test=score_subset(data,np.flatnonzero(folds==k),beta,cov,args.min_separation,efficiencies)
            calibration=fit_calibrator(cal.log_overlap.to_numpy(),cal.same_family.to_numpy())
            test['probability']=probabilities(test.log_overlap.to_numpy(),calibration)
            # Operational prefilter is part of measured recall: rejected pairs
            # cannot appear in screening output, regardless of calibration score.
            test['screened_in']=test.min_mahalanobis2<=args.q_limit
            test['proposal_probability']=np.where(test.screened_in,test.probability,0.)
            test['fold']=k;cal['fold']=k
            test.to_csv(directory/'test_pairs.csv',index=False);cal.to_csv(directory/'calibration_pairs.csv',index=False)
            write_json(directory/'calibrator.json',calibration)
            entry=dict(fold=k,complete=True,training_images=int(training.sum()),calibration_images=int((folds==calibration_fold).sum()),
                test_images=int((folds==k).sum()),fit=fit,field_timing=field_time,seconds=time.perf_counter()-begin,
                calibration_metrics=metrics(test.same_family,test.probability),
                screening_metrics=metrics(test.same_family,test.proposal_probability))
            tests.append(test);calibrations.append(cal)
        except Exception as error:
            import traceback
            (directory/'failure.txt').write_text(traceback.format_exc())
            entry=dict(fold=k,complete=False,error=repr(error),seconds=time.perf_counter()-begin)
        status.append(entry);write_json(directory/'status.json',entry);write_json(output/'fold_status.json',status)
    if len(tests)<3:raise RuntimeError('Fewer than three successful folds; no deployable probability calibration')
    heldout=pd.concat(tests,ignore_index=True);training_cal=pd.concat(calibrations,ignore_index=True)
    heldout.to_csv(output/'heldout_pairs.csv',index=False)
    final=fit_calibrator(training_cal.log_overlap.to_numpy(),training_cal.same_family.to_numpy())
    write_json(output/'calibration.json',dict(signature=key,settings=settings,calibration=final,efficiencies=efficiencies,
        scope='P(same catalog family | geometric score), calibrated on known multiply-imaged knots; not an unconditional field-catalog probability',
        successful_folds=len(tests),total_folds=args.folds,
        limitation='No colors, morphology, image existence or posterior lens uncertainty; probabilities require transport validation for other catalogs.'))
    overall=metrics(heldout.same_family,heldout.probability)
    screening=metrics(heldout.same_family,heldout.proposal_probability)
    write_json(output/'metrics.json',dict(calibration=overall,screening=screening,folds=status))
    make_report(output,status,overall,screening)
    print(json.dumps(dict(successful_folds=len(tests),metrics=screening),indent=2))

def make_report(output,status,calibration,screening):
    from feasibility.common import plot_style
    lines=['# Fast family-screening validation','',
        'Whole redshift groups are held out, keeping equal-redshift knots together. Family 0141 and its redshift group remain training-only.',
        'Each fit uses separate lens-training, probability-calibration and test groups. Test associations never train their lens or calibrator.',
        'The selected member catalog is held fixed. It was selected using all known image positions, without using their family labels.',
        'Probabilities apply to the known multiple-image/knot candidate population. They are not calibrated for arbitrary field galaxies.',
        'Every eligible pair is evaluated in the tests. Shared-image pairs are dependent; pair counts are not independent validation sample sizes.',
        '',f"Completed folds: {sum(s.get('complete',False) for s in status)}/{len(status)}.",
        f"Held-out pairs: {calibration['pairs']}; true pairs: {calibration['positives']}; Brier score: {calibration['brier']:.5g}; average precision: {calibration['average_precision']:.5g}.",
        '', '| Probability threshold | Proposals | Precision | Recall |','|---|---:|---:|---:|']
    for row in screening['thresholds']:
        prec='no proposals' if row['precision'] is None else f"{row['precision']:.3f}"
        rec='unavailable' if row['recall'] is None else f"{row['recall']:.3f}"
        lines.append(f"| {row['probability']} | {row['proposals']} | {prec} | {rec} |")
    lines+=['','Recall includes the source-distance prefilter but is conditional on the minimum image separation and redshift grid.',
        'These are pair probabilities, not transitive family assignments. A-B and B-C matches do not establish A-C or a joint three-image family.',
        'Full results and failure/optimizer diagnostics are in fold_status.json, metrics.json and heldout_pairs.csv.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    try:
        with plot_style() as plt:
            fig,axes=plt.subplots(1,2,figsize=(7,3))
            bins=calibration['reliability'];axes[0].plot([0,1],[0,1],color='gray',ls=':')
            axes[0].plot([b['mean_probability'] for b in bins],[b['observed_fraction'] for b in bins],'o-',color='tab:blue')
            axes[0].set(xlabel='Predicted probability',ylabel='Observed same-family fraction',xlim=(0,1),ylim=(0,1))
            rows=[r for r in screening['thresholds'] if r['precision'] is not None]
            axes[1].plot([r['recall'] for r in rows],[r['precision'] for r in rows],'o-',color='tab:blue')
            axes[1].set(xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1))
            fig.tight_layout();fig.savefig(output/'validation.pdf');plt.close(fig)
    except Exception as error:(output/'plot_failure.txt').write_text(repr(error))

def screen(args):
    import numpy as np
    import pandas as pd
    from feasibility.model import Model
    from feasibility.common import read_json,write_json
    from feasibility.catalogs import REFERENCE
    from lenscluster.inputs import radec_to_offsets_arcsec
    from feasibility.family_probability import geometry,candidate_pairs,pair_scores,probabilities
    destination=args.output/'screened_pairs.csv'
    if destination.exists() and not args.overwrite:raise ValueError('screened_pairs.csv exists; use --overwrite to replace screening output only')
    bundle=read_json(args.output/'calibration.json')
    if bundle is None:raise ValueError('Run validate first to build empirical calibration')
    settings=bundle['settings']
    if signature(settings)!=bundle['signature']:raise ValueError('Calibration inputs/code changed: revalidate in a new output directory')
    if args.catalog:
        data=pd.read_csv(args.catalog,dtype={'id':str})
        if not {'id','ra','dec'}<=set(data):raise ValueError('CSV needs id, ra, dec columns (degrees)')
        if ((data.ra<0)|(data.ra>=360)|(data.dec< -90)|(data.dec>90)).any():raise ValueError('Invalid RA/Dec in degrees')
        if args.pair_prior is None:raise ValueError('For a new catalog provide --pair-prior: expected same-family fraction among all eligible unordered pairs. This is an assumption, not inferred from the lens.')
        data['x'],data['y']=radec_to_offsets_arcsec(data.ra.to_numpy(),data.dec.to_numpy(),*REFERENCE[1:])
    else:data=load_known()
    if len(data)<2:raise ValueError('Screening needs at least two images')
    xy=data[['x','y']].to_numpy(float)
    if data.id.isna().any() or data.id.duplicated().any() or not np.isfinite(xy).all():raise ValueError('Missing/duplicate IDs or nonfinite coordinates')
    model=Model();best=np.load(ROOT/'feasibility/baseline_map.npz')
    if list(best['names'])!=model.names:raise ValueError('Saved best-fit parameter names differ from the current model')
    physical=model.ev.physical(model.full(best['u']));begin=time.perf_counter()
    from scipy.spatial import Delaunay
    outside=Delaunay(model.xy).find_simplex(xy)<0
    beta,cov,timing=geometry(model,physical,xy,bundle['efficiencies'],settings['sigma_obs'])
    pairs=candidate_pairs(beta,cov,xy,settings['min_separation'],settings['q_limit'],args.max_pairs)
    score,q,bestf=pair_scores(beta,cov,pairs)
    p=probabilities(score,bundle['calibration'],args.pair_prior)
    keep=(q<=settings['q_limit'])&(p>=args.min_probability)
    pairs,score,q,p,bestf=pairs[keep],score[keep],q[keep],p[keep],bestf[keep]
    i,j=pairs.T
    result=pd.DataFrame(dict(id_a=data.id.to_numpy()[i],id_b=data.id.to_numpy()[j],probability=p,
        log_overlap=score,min_mahalanobis2=q,best_efficiency=np.asarray(bundle['efficiencies'])[bestf],
        separation_arcsec=np.linalg.norm(xy[i]-xy[j],axis=1),outside_constraint_footprint=outside[i]|outside[j]))
    result=result.sort_values('probability',ascending=False,kind='stable')
    result.to_csv(destination,index=False)
    write_json(args.output/'screening_status.json',dict(images=len(data),pairs_written=len(result),seconds=time.perf_counter()-begin,
        field_timing=timing,pair_prior=args.pair_prior if args.pair_prior is not None else bundle['calibration']['reference_pair_prior'],
        minimum_probability=args.min_probability,catalog=str(args.catalog) if args.catalog else 'known constraint catalog (in-sample screening)',
        input_sha256=hashlib.sha256(args.catalog.read_bytes()).hexdigest() if args.catalog else None,
        outside_constraint_footprint=int(outside.sum()),
        probability_scope=bundle['scope'],warning='Prior correction assumes only the positive-pair prevalence changes. Field galaxies may violate that assumption; probabilities are provisional.'))
    print(result.head(25).to_string(index=False));print('Saved',destination)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cores',type=int,default=None)
    parser.add_argument('--no-progress',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'feasibility/family_screening')
    commands=parser.add_subparsers(dest='command',required=True)
    validation=commands.add_parser('validate',help='Fast whole-family held-out fits and probability calibration')
    validation.add_argument('--folds',type=int,default=5);validation.add_argument('--seed',type=int,default=20261101)
    validation.add_argument('--maxiter',type=int,default=800);validation.add_argument('--resume',action='store_true')
    validation.add_argument('--z-grid',type=int,default=64);validation.add_argument('--z-min',type=float,default=.6)
    validation.add_argument('--z-max',type=float,default=10.);validation.set_defaults(sigma_obs=.3)
    validation.add_argument('--min-separation',type=float,default=0.);validation.add_argument('--q-limit',type=float,default=36.)
    screening=commands.add_parser('screen',help='Score candidate image pairs using the saved model and calibration')
    screening.add_argument('--catalog',type=Path);screening.add_argument('--pair-prior',type=float)
    screening.add_argument('--max-pairs',type=int,default=500000);screening.add_argument('--min-probability',type=float,default=.1)
    screening.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    from feasibility.runtime import configure
    configure(args.cores,1,not args.no_progress)
    if args.command=='validate':
        if args.folds<3 or args.maxiter<1 or args.z_grid<2 or not .396<args.z_min<args.z_max:
            parser.error('Need folds>=3, maxiter>=1, z-grid>=2, and .396<z-min<z-max')
        if args.sigma_obs<=0 or args.min_separation<0 or args.q_limit<=0:parser.error('Invalid error, separation or Q threshold')
        settings={k:getattr(args,k) for k in ('folds','seed','maxiter','z_grid','z_min','z_max','sigma_obs','min_separation','q_limit')}
        validate(args,settings)
    else:
        if not 0<=args.min_probability<=1 or args.max_pairs<1:parser.error('Invalid probability or candidate limit')
        if args.pair_prior is not None and not 0<args.pair_prior<1:parser.error('--pair-prior must be strictly between zero and one')
        screen(args)

if __name__=='__main__':main()
