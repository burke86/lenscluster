"""Saved-draw member-shape diagnostics. No inference; angles use the catalog coordinate convention."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from scipy.stats import chi2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lenscluster.inputs import lenstool_shape_to_e1e2
from lenscluster.plot_style import PAPER_STYLE

BLUE='#0072B2';ORANGE='#D55E00';GRAY='.65';CHAIN=['#0072B2','#D55E00','#009E73','#CC79A7']
def wrap(a):return (a+90)%180-90

def quantities(e,reference):
 m=np.linalg.norm(e,axis=-1)
 return 2*m/(1+m*m),wrap(np.degrees(np.arctan2(e[...,1],e[...,0]))/2-reference)

def ellipse(ax,mean,cov,color,ls='-',lw=1.):
 eig,vec=np.linalg.eigh(cov);order=np.argsort(eig)[::-1];eig=eig[order];vec=vec[:,order]
 axes=2*np.sqrt(chi2.ppf(.68,2)*eig)
 ax.add_patch(Ellipse(mean,*axes,angle=np.degrees(np.arctan2(vec[1,0],vec[0,0])),fill=False,color=color,ls=ls,lw=lw))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();run=a.run.resolve();final=run/'final'
 state=json.loads((run/'init/artifacts/state.json').read_text());cfg=json.loads((final/'artifacts/config.json').read_text());diag=json.loads((final/'artifacts/posterior_diagnostics.json').read_text())
 specs=state['parameter_specs'];names=[s['name']for s in specs];scatter=cfg['scaling']['member_shape_scatter'];assert scatter>0
 members=pd.read_csv(final/'tables/member_diagnostics.csv',dtype={'catalog_id':str,'member_id':str})
 assert members.member_id.tolist()==state['member_ids']
 with np.load(final/'artifacts/posterior.npz',allow_pickle=True)as saved:
  physical=saved['physical'];latent=saved['samples'];chains=saved['grouped_samples'];divergences=int(saved['diverging'].sum())
  assert chains.ndim==3 and chains.shape[0]*chains.shape[1]==len(physical)
  np.testing.assert_allclose(chains.reshape(latent.shape),latent)
 indices=np.array([[names.index(f'{r.potfile_id}.{r.catalog_id}.shape_e{k}_offset')for k in (1,2)]for r in members.itertuples()])
 for i in indices.ravel():assert specs[i]['prior_kind']=='normal'and specs[i]['mean']==0 and specs[i]['std']==1 and specs[i]['transform_kind']=='identity'
 offset=physical[:,indices];np.testing.assert_allclose(offset,latent[:,indices])
 catalog_pa=members.angle_pos.fillna(0).to_numpy()
 pa_defined=members.angle_pos.notna().to_numpy() & (members.ellipticite.to_numpy()>1e-10)
 base=np.array([lenstool_shape_to_e1e2(r.ellipticite,angle)for r,angle in zip(members.itertuples(),catalog_pa)])
 active=np.zeros(len(members),bool);active[np.asarray(state['exact_positions'],int)]=True
 np.testing.assert_array_equal(active,members.member_class.eq('exact'))
 reconstructed=base[None]+scatter*offset
 assert np.all(np.linalg.norm(reconstructed[:,active],axis=-1)<1)
 eps,delta_pa=quantities(reconstructed,catalog_pa[None])
 # Verify the catalog convention round-trip, including the 180-degree angle periodicity.
 e0,pa0=quantities(base,catalog_pa);np.testing.assert_allclose(e0,members.ellipticite,atol=1e-12)
 np.testing.assert_allclose(pa0[np.linalg.norm(base,axis=1)>1e-8],0,atol=1e-10)
 eq=np.quantile(eps,[.16,.5,.84],axis=0);pq=np.quantile(delta_pa,[.16,.5,.84],axis=0)
 rng=np.random.default_rng(4106);prior_unit=rng.normal(size=(60000,2));prior_eq=[];prior_pq=[];prior_cov=[];reject=[];prior_resultant=[]
 for j in range(len(members)):
  e=base[j]+scatter*prior_unit;keep=np.linalg.norm(e,axis=1)<1 if active[j] else np.ones(len(e),bool)
  reject.append(float(1-keep.mean()));ep,pa=quantities(e[keep],catalog_pa[j]);prior_resultant.append(float(abs(np.mean(np.exp(2j*np.deg2rad(pa))))));prior_eq.append(np.quantile(ep,[.16,.5,.84]));prior_pq.append(np.quantile(pa,[.16,.5,.84]));prior_cov.append(np.cov(prior_unit[keep].T) if not keep.all() else np.eye(2))
 prior_eq=np.array(prior_eq).T;prior_pq=np.array(prior_pq).T;prior_cov=np.array(prior_cov)
 cov=np.array([np.cov(offset[:,j,:].T)for j in range(len(members))]);means=offset.mean(0)
 ratio=(np.linalg.det(cov)/np.linalg.det(prior_cov))**.25
 shifts=np.sqrt(np.einsum('ni,nij,nj->n',means,np.linalg.inv(prior_cov),means))
 rh=np.max(np.asarray(diag['r_hat'])[indices],axis=1);ess=np.min(np.asarray(diag['n_eff'])[indices],axis=1)
 resultant=np.abs(np.mean(np.exp(2j*np.deg2rad(delta_pa)),axis=0));weak=resultant<.7
 reliable=(rh<1.05)&(ess>=100)
 # Prior-only draws with the same sample count give a finite-sample width baseline, not a convergence correction.
 null=rng.normal(size=(3000,len(physical),2));c=np.einsum('bni,bnj->bij',null-null.mean(1)[:,None],null-null.mean(1)[:,None])/(len(physical)-1)
 null_limits=np.quantile(np.linalg.det(c)**.25,[.025,.975])
 summary=members[['member_id','catalog_id','catalog_mag','member_class','ellipticite','angle_pos','nearest_image_arcsec','x_centre','y_centre']].copy()
 for name,value in [('shape_active',active),('width_ratio',ratio),('mean_shift_prior_units',shifts),('shape_rhat_max',rh),('shape_ess_min',ess),('shape_diagnostics_pass',reliable),('pa_resultant',resultant),('pa_weak',weak),('prior_rejection_fraction',reject)]:summary[name]=value
 for name,qs in [('ellipticity',eq),('delta_pa_deg',pq),('prior_ellipticity',prior_eq),('prior_delta_pa_deg',prior_pq)]:
  for suffix,v in zip(['p16','median','p84'],qs):summary[f'{name}_{suffix}']=v
 # Far-field coordinates are unused latent variables, not recovered physical galaxy shapes.
 for col in ['ellipticity_p16','ellipticity_median','ellipticity_p84','delta_pa_deg_p16','delta_pa_deg_median','delta_pa_deg_p84']:
  summary.loc[~active,col]=np.nan
 prefix=cfg['paths']['figure_prefix'];figures=final/'figures';tables=final/'tables';out=figures/f'{prefix}_member_shape'
 for col in ['delta_pa_deg_p16','delta_pa_deg_median','delta_pa_deg_p84','prior_delta_pa_deg_p16','prior_delta_pa_deg_median','prior_delta_pa_deg_p84']:
  summary.loc[~pa_defined,col]=np.nan
 cart_q=np.quantile(reconstructed,[.16,.5,.84],axis=0)
 summary['prior_pa_resultant']=prior_resultant
 for k in range(2):
  summary[f'e{k+1}_catalog']=base[:,k]
  for qi,suffix in enumerate(['p16','median','p84']):
   summary[f'e{k+1}_posterior_{suffix}']=np.where(active,cart_q[qi,:,k],np.nan)
 summary.to_csv(tables/'member_shape_recovery.csv',index=False)
 np.savez_compressed(tables/'member_shape_recovery_draws.npz',unit_offsets=offset,base_e1e2=base,active=active,ellipticity=eps[:,active],delta_pa_deg=delta_pa[:,active],active_member_ids=members.member_id[active].to_numpy(dtype=str))
 plt.style.use(Path(__file__).resolve().parents[1]/'src/lenscluster/style.mplstyle');plt.rcParams.update(PAPER_STYLE)
 def finish(fig,path):
  for ax in fig.axes:ax.grid(False);ax.tick_params(which='both',top=True,right=True)
  fig.savefig(path,dpi=300,bbox_inches='tight',pad_inches=.04);plt.close(fig);print(path,flush=True)
 def bars(ax,x,q,mask,color,label=None):
  return ax.errorbar(np.asarray(x)[mask],q[1,mask],yerr=np.maximum(0,np.stack((q[1]-q[0],q[2]-q[1]))[:,mask]),fmt='o',ms=3,mew=.5,elinewidth=.45,capsize=0,color=color,alpha=.7,label=label)
 fig,ax=plt.subplots(2,2,figsize=(7.1,6.4),layout='constrained')
 good=active&reliable;bad=active&~reliable
 for mask,col,label in [(good,BLUE,'Shape diagnostics pass'),(bad,ORANGE,'Shape diagnostics fail')]:
  bars(ax[0,0],members.ellipticite,eq,mask,col,label)
 ax[0,0].plot([0,1],[0,1],color='.35',ls='--',lw=.7);ax[0,0].set(xlabel=r'$\epsilon_{\rm cat}$',ylabel=r'$\epsilon_{\rm post}$',xlim=(0,1),ylim=(0,1));ax[0,0].legend(fontsize=6.5,frameon=False)
 cat_pa=wrap(catalog_pa);post_pa=cat_pa[None]+pq
 bars(ax[0,1],cat_pa,post_pa,active&pa_defined&~weak,BLUE,'PA concentrated');bars(ax[0,1],cat_pa,post_pa,active&pa_defined&weak,GRAY,'PA broad')
 ax[0,1].plot([-90,90],[-90,90],color='.35',ls='--',lw=.7);ax[0,1].set(xlabel=r'$\phi_{\rm cat}$ (deg)',ylabel=r'$\phi_{\rm post}$ (deg)',xlim=(-95,95));ax[0,1].legend(fontsize=6.5,frameon=False)
 # Compare posterior summaries to the transformed prior, not only to the input light shape.
 bars(ax[1,0],members.ellipticite,eq-prior_eq[1][None],good,BLUE);bars(ax[1,0],members.ellipticite,eq-prior_eq[1][None],bad,ORANGE)
 ax[1,0].axhline(0,color='.35',ls='--',lw=.7);ax[1,0].set(xlabel=r'$\epsilon_{\rm cat}$',ylabel=r'$\epsilon_{\rm post}-\mathrm{median}(\epsilon_{\rm prior})$')
 for mask,col in [(good,BLUE),(bad,ORANGE)]:ax[1,1].scatter(shifts[mask],ratio[mask],s=13,color=col)
 ax[1,1].axhline(1,color='.35',ls='--',lw=.7);ax[1,1].set(xlabel=r'$|\langle\Delta\boldsymbol{e}\rangle|/\sigma_e$',ylabel=r'$s_{\rm post}/s_{\rm prior}$')
 strongest=np.flatnonzero(active)[np.argsort(ratio[active])[:5]]
 offsets={'8971':(-5,7),'1297':(-9,-12),'7918':(6,7),'8514':(-6,7),'4112':(7,-5)}
 for j in strongest:
  dx,dy=offsets.get(members.catalog_id.iloc[j],(3,3))
  ax[1,1].annotate(members.catalog_id.iloc[j],(shifts[j],ratio[j]),xytext=(dx,dy),textcoords='offset points',fontsize=6,ha='right' if dx<0 else 'left',arrowprops=dict(arrowstyle='-',color='.4',lw=.4))
 finish(fig,str(out)+'_comparison.pdf')
 # Preferred physical comparison: Cartesian shapes and the nonlinear transformed prior.
 fig,axes=plt.subplots(2,2,figsize=(7.1,6.2),layout='constrained')
 for k,ax in enumerate(axes[0]):
  q=cart_q[:,:,k]
  extent=np.array([min(base[active,k].min(),q[0,active].min())-.025,max(base[active,k].max(),q[2,active].max())+.025])
  ax.fill_between(extent,extent-scatter,extent+scatter,color='.85',alpha=.5)
  ax.plot(extent,extent,color='.3',ls='--',lw=.7,label='Prior centre')
  for mask,col,label in [(good,BLUE,'Shape diagnostics pass'),(bad,ORANGE,'Shape diagnostics fail')]:bars(ax,base[:,k],q,mask,col,label)
  ax.set(xlim=extent,ylim=extent,xlabel=rf'$e_{{{k+1},\rm cat}}$',ylabel=rf'$e_{{{k+1},\rm post}}$')
 axes[0,0].legend(frameon=False,fontsize=6.5,loc='upper left')
 ax=axes[1,0]
 for mask,col,label in [(good,BLUE,None),(bad,ORANGE,None)]:bars(ax,prior_eq[1],eq,mask,col,label)
 ax.plot([0,1],[0,1],color='.3',ls='--',lw=.7)
 ax.set(xlabel=r'$\mathrm{median}(\epsilon_{\rm prior})$',ylabel=r'$\mathrm{median}(\epsilon_{\rm post})$',xlim=(0,1),ylim=(0,1))
 ax=axes[1,1]
 for mask,col in [(good,BLUE),(bad,ORANGE)]:ax.scatter(np.asarray(prior_resultant)[mask],resultant[mask],s=15,color=col)
 ax.plot([0,1],[0,1],color='.3',ls='--',lw=.7)
 ax.set(xlabel=r'$R_{\phi,\rm prior}$',ylabel=r'$R_{\phi,\rm post}$',xlim=(0,1),ylim=(0,1))
 ax.text(.04,.94,'0: broad angle; 1: concentrated',transform=ax.transAxes,va='top',fontsize=7)
 finish(fig,str(out)+'_prior_comparison.pdf')

 fig,ax=plt.subplots(2,2,figsize=(7.1,6.3),layout='constrained')
 bins=np.linspace(min(.5,ratio.min()-.03),max(1.2,ratio.max()+.03),22)
 ax[0,0].hist(ratio[active],bins=bins,histtype='step',density=True,color=BLUE,label=f'Active ({active.sum()})');ax[0,0].hist(ratio[~active],bins=bins,histtype='step',density=True,color=ORANGE,label=f'Inactive ({(~active).sum()})')
 ax[0,0].axvspan(*null_limits,color='.8',alpha=.3);ax[0,0].axvline(1,color='.3',ls='--',lw=.7);ax[0,0].set(xlabel=r'$s_{\rm post}/s_{\rm prior}$',ylabel='Density');ax[0,0].legend(fontsize=7,frameon=False)
 sc=ax[0,1].scatter(members.nearest_image_arcsec[active],ratio[active],c=members.catalog_mag[active],cmap='viridis',s=15)
 ax[0,1].set(xscale='log',xlabel=r'$d_{\rm image}$ (arcsec)',ylabel=r'$s_{\rm post}/s_{\rm prior}$');ax[0,1].axhline(1,color='.3',ls='--',lw=.7);fig.colorbar(sc,ax=ax[0,1],label=r'$m_{\rm cat}$')
 for mask,col,label in [(active,BLUE,'Active'),(~active,ORANGE,'Inactive')]:ax[1,0].scatter(ratio[mask],rh[mask],s=12,color=col,label=label)
 ax[1,0].axhline(1.05,color='.35',ls='--',lw=.7);ax[1,0].set(xlabel=r'$s_{\rm post}/s_{\rm prior}$',ylabel=r'$\max(\widehat R_{e_1},\widehat R_{e_2})$');ax[1,0].legend(fontsize=7,frameon=False)
 from cmcrameri import cm
 sc=ax[1,1].scatter(members.x_centre[active],members.y_centre[active],c=ratio[active],cmap=cm.batlow_r,s=22,vmin=min(.6,ratio[active].min()),vmax=1.1)
 images=pd.read_csv(tables/'image_fit_quality.csv');ax[1,1].scatter(images.x_obs_arcsec,images.y_obs_arcsec,s=3,c='.6',marker='x',linewidths=.3)
 ax[1,1].set(xlabel=r'$x$ (arcsec)',ylabel=r'$y$ (arcsec)',aspect='equal');fig.colorbar(sc,ax=ax[1,1],label=r'$s_{\rm post}/s_{\rm prior}$')
 finish(fig,str(out)+'_constraints.pdf')
 selected=list(strongest[:3])
 for j in np.flatnonzero(active)[np.argsort(shifts[active])[::-1]]:
  if j not in selected:selected.append(j)
  if len(selected)==6:break
 fig,axes=plt.subplots(2,3,figsize=(7.1,5.1),layout='constrained',sharex=True,sharey=True)
 for ax,j in zip(axes.flat,selected):
  for ch in range(chains.shape[0]):
   pts=chains[ch,:,indices[j]].T
   ax.scatter(pts[:,0],pts[:,1],s=3,color=CHAIN[ch],alpha=.25,rasterized=True)
   ax.plot(*pts.mean(0),'x',color=CHAIN[ch],ms=5,mew=1.)
  ellipse(ax,[0,0],prior_cov[j],'.3','--',.8);ellipse(ax,means[j],cov[j],BLUE,'-',1.)
  ax.text(.04,.96,f'{members.catalog_id.iloc[j]}\n'+rf'$s_{{\rm post}}/s_{{\rm prior}}={ratio[j]:.2f}$'+'\n'+rf'$\widehat R_{{\max}}={rh[j]:.2f}$',transform=ax.transAxes,va='top',fontsize=7,bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=1.5))
  limit=max(3.5,np.ceil(np.max(np.abs(offset[:,selected]))*2)/2)
  ax.set(xlim=(-limit,limit),ylim=(-limit,limit),aspect='equal')
 for ax in axes[-1]:ax.set_xlabel(r'$\Delta e_1/\sigma_e$')
 for ax in axes[:,0]:ax.set_ylabel(r'$\Delta e_2/\sigma_e$')
 axes[0,2].legend(handles=[Line2D([],[],color='.3',ls='--',label='Prior'),Line2D([],[],color=BLUE,label='Posterior')],loc='lower right',fontsize=6,frameon=False)
 fig.legend(handles=[Line2D([],[],color=CHAIN[ch],marker='x',ls='none',label=f'Chain {ch+1} mean')for ch in range(chains.shape[0])],loc='lower center',bbox_to_anchor=(.5,-.045),ncol=4,frameon=False,fontsize=7)
 finish(fig,str(out)+'_selected_posteriors.pdf')
 report=dict(run=str(run),retained_draws=len(physical),chains=int(chains.shape[0]),draws_per_chain=int(chains.shape[1]),active=int(active.sum()),active_circular_catalog=int((active&~pa_defined).sum()),inactive=int((~active).sum()),sigma_e=scatter,epsilon_definition='(a^2-b^2)/(a^2+b^2) = 2|e|/(1+|e|^2)',angle_definition='catalog angle convention, axial modulo 180 deg; posterior unwrapped around catalog PA',width_definition='s_post/s_prior = (det Cov_post(e1,e2)/det Cov_prior(e1,e2))^(1/4); a geometric-mean linear width, not an area ratio',prior='independent Normal(0,1) offsets, rejecting |e|>=1 for active shapes; 60000 seeded prior draws per member for nonlinear transforms',shape_diagnostics='pass iff both coordinate R-hat<1.05 and min ESS>=100; passing does not certify global convergence',active_pass=int((active&reliable).sum()),active_width_median=float(np.median(ratio[active])),inactive_width_median=float(np.median(ratio[~active])),active_width_below_08=int((active&(ratio<.8)).sum()),active_width_below_08_pass=int((active&(ratio<.8)&reliable).sum()),active_shift_above1=int((active&(shifts>1)).sum()),active_max_rhat=float(rh[active].max()),global_max_rhat=diag['max_r_hat'],divergences=divergences,selected_member_ids=members.catalog_id.iloc[selected].tolist(),null_width_95percent_interval=null_limits.tolist(),interpretation='Exploratory short-run diagnostics; inactive shape offsets are prior-only nuisance parameters, not mass-shape measurements. Individual width changes can reflect finite sampling or poor mixing. Ellipses show Gaussian covariance summaries, not measured credible contours; colors denote chains and crosses their means.')
 (tables/'member_shape_recovery.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
