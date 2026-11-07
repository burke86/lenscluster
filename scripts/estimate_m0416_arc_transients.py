"""Full-arc magnification-selected detection proxy, not a microlensing simulation."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, cKDTree
from scipy.ndimage import distance_transform_edt
import contourpy

RUN='results/nov3o_densemass_notfast/m0416_canucs2025/w500s250t8_gamma4_allblocks'
OUT=Path('output/pdf/m0416_arc_transients')
SCALE=5.340068651570361

def maps(run, system, step, out):
    from _covariance_plot_cli import load_run
    config,state,evaluator,posterior,paths=load_run(run)
    from lenscluster.fold_diagnostics import load_fold_data
    data=load_fold_data(paths.tables)
    s=next(s for s in data['systems'] if s['system_id']==system)
    knots=np.asarray(s['observed'])
    lo,hi=knots.min(0)-1.,knots.max(0)+1.
    x=np.arange(lo[0],hi[0]+step/2,step); y=np.arange(lo[1],hi[1]+step/2,step)
    xx,yy=np.meshgrid(x,y); points=np.column_stack([xx.ravel(),yy.ravel()])
    best=int(np.nanargmax(posterior.log_prob)); theta=posterior.samples[best]
    physical=evaluator.physical(theta)
    packed=evaluator.packed_state_for_z(physical,float(s['z_source']))
    print(f'{system}: {len(points)} grid points, best draw {best}',flush=True)
    _,h=evaluator.total_fields_at(points,packed,chunk=1024)
    correction=evaluator.gp_deflection_correction(theta)
    family=list(map(str,state.images.family_ids)).index(str(s['anchor_family_id']))
    grf=np.empty_like(h)
    for start in range(0,len(points),384):
        q=points[start:start+384]; n=len(q)
        q=np.pad(q,((0,384-n),(0,0)),mode='edge')
        c=correction(q[:,0],q[:,1],np.full(384,family))
        grf[start:start+n]=np.stack(c[2:],axis=-1)[:n].reshape(n,2,2)
        if start % (384*20)==0: print(f'GRF {start}/{len(points)}',flush=True)
    a=np.eye(2)-h; b=a-grf
    dest=out/f'{system}_map_{step:g}.npz'
    np.savez_compressed(dest,x=x,y=y,knots=knots,det_parametric=np.linalg.det(a).reshape(xx.shape),
        det_grf=np.linalg.det(b).reshape(xx.shape),best_draw=best,z_source=s['z_source'],
        jac_parametric=a.reshape(*xx.shape,2,2),jac_grf=b.reshape(*xx.shape,2,2))
    return dest

def distribution(offset,weight,bins):
    w=weight.ravel();d=offset.ravel();order=np.argsort(d);v=d[order];cdf=np.cumsum(w[order])/w.sum()
    q=np.interp([.05,.5,.95],cdf,v)
    return np.histogram(d,bins,weights=w)[0]/w.sum()/np.diff(bins),dict(
        mean_kpc=float(np.sum(w*d)/w.sum()),median_kpc=float(q[1]),width90_kpc=float(q[2]-q[0]),
        q05_kpc=float(q[0]),q95_kpc=float(q[2]),negative_reference_fraction=float(w[d<0].sum()/w.sum()))

def analyze(path,out,system):
    d=np.load(path);x,y=d['x'],d['y'];xx,yy=np.meshgrid(x,y);points=np.column_stack([xx.ravel(),yy.ravel()])
    step=x[1]-x[0];knots=d['knots'];hp=ConvexHull(knots).equations
    inside=np.all(points@hp[:,:2].T+hp[:,2]<=1e-12,axis=1).reshape(xx.shape)
    distance=distance_transform_edt(~inside,sampling=(step,step))
    mask=inside|(distance<=.35)
    dp,dg=d['det_parametric'],d['det_grf']
    curves=contourpy.contour_generator(x=x,y=y,z=dp).lines(0.)
    # Densify line segments for accurate nearest-curve Euclidean distances.
    nodes=[]
    for c in curves:
        for a,b in zip(c[:-1],c[1:]):
            nodes.append(a+np.linspace(0,1,max(2,int(np.ceil(np.linalg.norm(b-a)/(step/5)))+1))[:,None]*(b-a))
    dist=cKDTree(np.vstack(nodes)).query(points)[0].reshape(xx.shape)*SCALE
    offset=np.sign(dp)*dist
    extent=max(6,float(np.ceil(np.max(abs(offset[inside|(distance<=.5)])))))
    bins=np.arange(-extent,extent+.1,.2)
    colors=['#5c6670','#0072b2','#cc79a7','#d55e00'];records=[];weights=[]
    plt.rcParams.update({'font.family':'serif','font.size':11,'mathtext.fontset':'stix','axes.grid':False,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(7.2,4.8),layout='constrained')
    for i,(model,r) in enumerate([('parametric',1),('grf',1),('grf',1.5),('grf',2)]):
        det=dp if model=='parametric' else dg
        mu=1/np.maximum(abs(det),1e-12)
        w=mask*(mu>=100)*np.minimum(mu,1000)**.75*np.where(det<0,r,1)
        hist,stats=distribution(offset,w,bins)
        records.append(dict(model=model,negative_parity_factor=r,mu_min=100,mu_cap=1000,padding_arcsec=.35,**stats))
        label='Parametric, R=1' if model=='parametric' else f'Parametric + GRF, R={r:g}'
        ax.stairs(hist,bins,label=label,color=colors[i],lw=1.5)
        weights.append(w)
    ax.axvline(0,color='.6',lw=.7);ax.set_xlabel('Signed distance to parametric critical line (lens-plane kpc)')
    occupied=np.any(np.asarray(weights)>0,axis=0)
    ax.set_xlim(np.floor(offset[occupied].min())-.25,np.ceil(offset[occupied].max())+.25)
    ax.set_ylabel('Normalized detection proxy (kpc$^{-1}$)');ax.legend(frameon=False,fontsize=9)
    fig.get_layout_engine().set(rect=(0,.06,1,.94))
    fig.text(.5,.015,'Full K1 knot footprint; assumed selection weights; best draw; no mean subtraction.',ha='center',fontsize=9)
    fig.savefig(out/f'{system}_estimated_event_offsets.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.6),layout='constrained')
    for ax,w,label in zip(axes,[weights[0],weights[2]],['Parametric, R=1','Parametric + GRF, R=1.5']):
        z=np.where(mask,w/(w.sum()*step**2),np.nan)
        im=ax.pcolormesh(x,y,z,shading='auto',cmap='magma',rasterized=True)
        ax.contour(x,y,dp,levels=[0],colors=['#00bcd4'],linewidths=1)
        ax.contour(x,y,dg,levels=[0],colors=['#64e572'],linewidths=.7)
        ax.contour(x,y,mask.astype(float),levels=[.5],colors=['.5'],linewidths=.7)
        ax.scatter(*knots.T,s=15,facecolors='none',edgecolors='white',linewidths=.6)
        ax.text(.03,.97,label,transform=ax.transAxes,va='top',fontsize=9,bbox=dict(facecolor='white',edgecolor='none',alpha=.85))
        ax.set_aspect('equal');ax.set_xlabel('Model x (arcsec)');ax.set_ylabel('Model y (arcsec)')
        fig.colorbar(im,ax=ax,label='Normalized detection proxy (arcsec$^{-2}$)',shrink=.8)
    fig.get_layout_engine().set(rect=(0,.07,1,.93))
    fig.text(.5,.015,'Cyan: reference critical line. Green: GRF critical lines. Circles: all 12 K1 arc knots.',ha='center',fontsize=9)
    fig.savefig(out/f'{system}_arc_detection_map.pdf');plt.close(fig)
    # Selection sensitivity, retaining each entire arc footprint rather than fold cuts.
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained');sens=[]
    for threshold,ls in [(30,':'),(100,'-'),(300,'--')]:
        means=[];widths=[]
        for r in [1,1.25,1.5,1.75,2]:
            mu=1/np.maximum(abs(dg),1e-12)
            w=mask*(mu>=threshold)*np.minimum(mu,1000)**.75*np.where(dg<0,r,1)
            _,stats=distribution(offset,w,bins)
            sens.append(dict(mu_min=threshold,negative_parity_factor=r,**stats));means.append(stats['mean_kpc']);widths.append(stats['width90_kpc'])
        for ax,v in zip(axes,[means,widths]):ax.plot([1,1.25,1.5,1.75,2],v,ls=ls,marker='o',ms=3,label=rf'$|\mu|\geq {threshold}$')
    for ax,label in zip(axes,['Mean event-proxy offset (kpc)','Event-proxy 5-95% width (kpc)']):
        ax.set_xlabel('Assumed negative-parity detection factor R');ax.set_ylabel(label);ax.legend(frameon=False)
    fig.savefig(out/f'{system}_selection_sensitivity.pdf');plt.close(fig)
    aperture=[]
    for padding in [.2,.35,.5]:
        m=inside|(distance<=padding)
        for cap in [300,1000,3000]:
            mu=1/np.maximum(abs(dg),1e-12);w=m*(mu>=100)*np.minimum(mu,cap)**.75*np.where(dg<0,1.5,1)
            _,stats=distribution(offset,w,bins)
            aperture.append(dict(padding_arcsec=padding,mu_cap=cap,**stats))
    result=dict(system=system,map=str(path),best_draw=int(d['best_draw']),z_source=float(d['z_source']),
        grid_step_arcsec=float(step),arc_area_arcsec2=float(mask.sum()*step**2),knots=len(knots),
        formula='weight = arc_mask * I(abs(mu)>=mu_min) * min(abs(mu),mu_cap)**0.75 * (R if detA<0 else 1)',
        assumptions=['Uniform unlensed cumulative-count normalization over the footprint',
        'Count slope gamma=0.7 adopted from Dragon Arc, not measured for K1',
        'R is a free illustrative negative-parity enhancement, not fitted or simulated',
        'Macro-magnification selection is a proxy, not a microlensing detection probability',
        'Footprint = convex hull of all 12 catalogued a/b knots expanded by 0.35 arcsec, not image segmentation',
        'Reference is same-best-draw parametric detA=0; distances signed by reference detA',
        'Full GRF detA controls local parity weights; reference parity and local parity differ',
        'Best draw of unconverged chain; conditional-mean field only; no absolute event rates'],
        baseline=records,selection_sensitivity=sens,aperture_and_cap_sensitivity=aperture)
    (out/f'{system}_estimate.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out/f'{system}_event_proxy.npz',x=x,y=y,mask=mask,offset_kpc=offset,weights=np.asarray(weights),bins=bins)
    print(json.dumps(result['baseline'],indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',default=RUN);p.add_argument('--system',default='K1');p.add_argument('--step',type=float,default=.02);p.add_argument('--output',type=Path,default=OUT)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/f'{args.system}_map_{args.step:g}.npz'
    if not path.exists():path=maps(args.run,args.system,args.step,args.output)
    analyze(path,args.output,args.system)

if __name__=='__main__':main()
