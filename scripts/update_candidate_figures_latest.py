"""Refresh existing candidate figures using the default fixed M0416 lens."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import fit_arc_source as fas
from lenscluster.jax_cosmology import dpie_sigma0_factor_from_config
RUN=ROOT/'results/nov4b_densemass_notfast_parity/m0416_canucs2025/w2000s1000t8_gamma4_allblocks_shear'
BASE=ROOT/'output/apj_counterimages';OUT=BASE/'latest_model';OUT.mkdir(exist_ok=True);(OUT/'new_families').mkdir(exist_ok=True)
config,state,ev,post=fas.load_model(RUN)
idx=int(np.nanargmax(post.log_prob));theta=fas.jnp.asarray(post.samples[idx]);physical=ev.physical(theta)
mean=ev.gp_deflection_correction(theta);fids=list(map(str,state.images.family_ids));sources=np.asarray(ev.source_positions(theta));zs={str(f.family_id):float(f.z_source) for f in state.family_data}
assert state.gp_blocks=='all'
print('Loaded best sample',idx,flush=True)
def xy(ra,dec):return np.column_stack([-(np.asarray(ra)-state.reference[1])*3600*np.cos(np.deg2rad(state.reference[2])),(np.asarray(dec)-state.reference[2])*3600])
def fields(points,z):
 n=len(points);q=np.pad(points,((0,max(0,64-n)),(0,0)),mode='edge');packed=ev.packed_state_for_z(physical,float(z))
 a,h=ev.total_fields_at(q,packed);a=np.asarray(a).copy();h=np.asarray(h).copy()
 c=mean(q[:,0],q[:,1],np.zeros(len(q),dtype=int))
 weight=float(dpie_sigma0_factor_from_config(state.z_lens,z,state.cosmo_config))/float(dpie_sigma0_factor_from_config(state.z_lens,state.gp_reference_redshift,state.cosmo_config))/float(ev.gp_weights_per_family[0])
 a+=np.column_stack(c[:2])*weight;h+=np.stack(c[2:],axis=-1).reshape(-1,2,2)*weight
 return points-a[:n],np.eye(2)-h[:n]
def roots(start,z,beta):
 pos=np.array(start,copy=True)
 for _ in range(100):
  b,j=fields(pos,z);step=np.linalg.solve(j,(b-beta)[...,None])[...,0];step*=np.minimum(1,.25/np.maximum(np.linalg.norm(step,axis=1),1e-15))[:,None];pos-=step
  if np.max(np.linalg.norm(step,axis=1))<1e-8:break
 b,j=fields(pos,z);err=np.linalg.norm(b-beta,axis=1)
 return pos,err,1/np.linalg.det(j)
candidates=pd.read_csv(BASE/'candidate_catalogue.csv');individual=pd.read_csv(BASE/'individual_candidate_predictions.csv');audit=[]
for i,r in individual.iterrows():
 cand=candidates[candidates.phot_id==r.candidate_source].iloc[0];obs=xy([cand.RA],[cand.Dec]);fi=fids.index(r.family);z=zs[r.family]
 # Start at the old root and at the candidate to follow the local branch without a global search.
 start=np.vstack([[r.x_pred,r.y_pred],obs]);pos,err,mu=roots(start,z,sources[fi]);valid=np.flatnonzero(err<1e-4)
 if len(valid):
  # Preserve distinct previously tracked branches when their continuations remain valid.
  k=0 if 0 in valid else int(valid[0]);pred=pos[k];offset=float(np.linalg.norm(pred-obs[0]));m=float(mu[k])
 else:pred=np.array([np.nan,np.nan]);offset=np.nan;m=np.nan
 individual.loc[i,['x_pred','y_pred','mu_pred','offset_arcsec']]=[*pred,m,offset]
 audit.append(dict(candidate_source=int(r.candidate_source),family=r.family,source_equation_error=float(np.min(err)),recovered=bool(len(valid))))
 print(r.system,r.family,'offset',offset,'source error',np.min(err),flush=True)
individual.to_csv(OUT/'individual_candidate_predictions.csv',index=False)
for i,r in candidates.iterrows():
 q=individual[individual.candidate_source==r.phot_id].sort_values('offset_arcsec').iloc[0]
 for col in ['x_pred','y_pred','mu_pred','offset_arcsec']:candidates.loc[i,col]=q[col]
 candidates.loc[i,'prediction_family']=q.family
 _,j=fields(xy([r.RA],[r.Dec]),float(r.z_sys));candidates.loc[i,'mu']=1/np.linalg.det(j)[0]
candidates.to_csv(OUT/'candidate_catalogue.csv',index=False)
pd.DataFrame(audit).to_csv(OUT/'prediction_consistency_checks.csv',index=False)
fit=pd.read_csv(RUN/'final/tables/image_fit_quality.csv');fit.to_csv(OUT/'known_image_predictions.csv',index=False)
new=pd.read_csv(BASE/'new_families/new_family_catalogue.csv');new=new[new.system=='NF-A'].copy();obs=xy(new.RA,new.Dec)
# Fixed-lens geometric redshift: minimize unweighted backprojected source scatter.
# This is a diagnostic optimum, not a redshift posterior or an image-plane likelihood fit.
def objective(z):
 b,_=fields(obs,z);return float(np.sum((b-b.mean(axis=0))**2))
grid=np.linspace(.6,8,60);scores=np.array([objective(z) for z in grid]);k=int(np.argmin(scores));lo=grid[max(0,k-1)];hi=grid[min(len(grid)-1,k+1)]
opt=minimize_scalar(objective,bounds=(lo,hi),method='bounded',options={'xatol':1e-6});z=float(opt.x)
b,j=fields(obs,z);origin=b.mean(axis=0);pred,err,mu=roots(obs,z,origin);valid=err<1e-4
new['z_model']=z;new['z_sys']=z;new['z_type']='fixed-lens geometric diagnostic';new['mu']=1/np.linalg.det(j)
new['x_pred']=np.where(valid,pred[:,0],np.nan);new['y_pred']=np.where(valid,pred[:,1],np.nan);new['offset_arcsec']=np.where(valid,np.linalg.norm(pred-obs,axis=1),np.nan)
new.to_csv(OUT/'new_families/new_family_catalogue.csv',index=False)
pd.DataFrame(dict(ID=new.ID,beta_x=b[:,0],beta_y=b[:,1],source_x=origin[0],source_y=origin[1],source_scatter_arcsec=np.linalg.norm(b-origin,axis=1),root_error_arcsec=err)).to_csv(OUT/'new_families/source_consistency.csv',index=False)
print('Family A z',z,'source rms',np.sqrt(opt.fun/len(obs)),'image offsets',new.offset_arcsec.tolist(),flush=True)
meta=dict(run=str(RUN),best_sample_index=idx,posterior_sha256=hashlib.sha256((RUN/'final/artifacts/posterior.npz').read_bytes()).hexdigest(),reference=list(state.reference),model='parametric + conditional mean shared GRF, no image GP displacement',family_A_redshift_method='unweighted source-plane scatter minimum at fixed lens; no redshift uncertainty inferred',family_A_z=z,family_A_source_rms=float(np.sqrt(opt.fun/len(obs))),source_grid_redshifts=grid.tolist(),source_grid_scores=scores.tolist(),candidate_scope='existing seven candidates and family A only; no discovery or catalogue reclassification',sampling='none')
(OUT/'model_provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
# Reuse existing presentation, with all model-dependent inputs redirected to refreshed tables.
prefix=(BASE/'make_publication.py').read_text().split('# Reconcile catalogue positions')[0]
prefix=prefix.replace("OUT=ROOT/'output/apj_counterimages'; run=ROOT/'results/nov3o_densemass_notfast/m0416_canucs2025/w500s250t8_gamma4_allblocks'",f"OUT=ROOT/'output/apj_counterimages/latest_model'; run=Path({str(RUN)!r})")
(OUT/'make_publication.py').write_text(prefix)
renderer=(BASE/'apj_selected/make_figures.py').read_text().replace("Path('output/apj_counterimages/make_publication.py')","Path('output/apj_counterimages/latest_model/make_publication.py')")
renderer=renderer.replace("'Model z')","'Geometric z')")
renderer=renderer.replace("f'Offset {candidate.offset_arcsec:.2f}″'+(' (outside)' if outside else '')","(f'Offset {candidate.offset_arcsec:.2f}″'+(' (outside)' if outside else '')) if np.isfinite(candidate.offset_arcsec) else 'No local model root'")
renderer=renderer.replace('stamp(axs[row,i],rec,cols[i],(r.x_pred,r.y_pred))',"stamp(axs[row,i],rec,cols[i],(r.x_pred,r.y_pred));axs[row,i].text(.04,.07,f'Offset {r.offset_arcsec:.2f}″'+(' (outside)' if max(abs(r.x_pred-xy(r.RA,r.Dec)[0]),abs(r.y_pred-xy(r.RA,r.Dec)[1]))>.75 else ''),transform=axs[row,i].transAxes,color='white',fontsize=6,path_effects=stroke)")
renderer=renderer.replace("loc='lower left',facecolor='black',labelcolor='white',frameon=True,fontsize=5.5)","loc='upper left',bbox_to_anchor=(0,.87),facecolor='black',labelcolor='white',frameon=True,fontsize=5.5)")
(OUT/'make_figures.py').write_text(renderer)
print('Rendering refreshed figures',flush=True)
exec(compile(renderer,str(OUT/'make_figures.py'),'exec'))
