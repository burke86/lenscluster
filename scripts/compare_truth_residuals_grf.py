"""Compare saved-run truth residuals with and without its shared conditional GRF."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.lines import Line2D
from _covariance_plot_cli import load_run
from lenscluster.truth_recovery import load_truth_fits, grid_arcsec
from lenscluster.jax_cosmology import dpie_sigma0_factor_from_config
from lenscluster.plot_style import PAPER_STYLE


def render(data, images, path):
    plt.style.use(Path(__file__).resolve().parents[1]/'src/lenscluster/style.mplstyle')
    plt.rcParams.update(PAPER_STYLE)
    cmap=LinearSegmentedColormap.from_list('cropped_RdBu',plt.get_cmap('RdBu_r')(np.linspace(.25,1,768)),N=768)
    cmap.set_bad('#e8e8e8');norm=Normalize(-1,2,clip=True)
    fig,axes=plt.subplots(2,3,figsize=(7.2,5.7),sharex=True,sharey=True)
    fig.subplots_adjust(left=.10,right=.99,bottom=.09,top=.86,wspace=.06,hspace=.06)
    x,y=data['x'],data['y'];meshes=[]
    for row,mode in enumerate(['parametric','with_grf']):
        for col,q in enumerate(['kappa','abs_mu']):
            truth=data[f'truth_{q}'];model=data[f'{mode}_{q}']
            residual=np.divide(model-truth,truth,out=np.full_like(truth,np.nan),where=np.isfinite(truth)&(truth>0)&np.isfinite(model))
            mesh=axes[row,col].pcolormesh(x,y,np.ma.masked_invalid(residual),cmap=cmap,norm=norm,shading='nearest',rasterized=True)
            if row==0:meshes.append(mesh)
        for key,color in [('truth_detA','#252525'),(f'{mode}_detA','#d55e00')]:
            axes[row,2].contour(x,y,data[key],levels=[0],colors=[color],linewidths=[.75])
        for ax in axes[row]:
            ax.scatter(images.x_obs_arcsec,images.y_obs_arcsec,s=4,facecolors='none',edgecolors='white',linewidths=.9,zorder=5)
            ax.scatter(images.x_obs_arcsec,images.y_obs_arcsec,s=4,facecolors='none',edgecolors='#333333',linewidths=.35,zorder=6)
            ax.set_aspect('equal');ax.tick_params(which='both',top=False,right=False);ax.grid(False)
        axes[row,0].set_ylabel('y (arcsec)')
        axes[row,0].text(.03,.97,['Parametric','Parametric + GRF'][row],transform=axes[row,0].transAxes,
                         va='top',fontsize=8,bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=2))
    for ax in axes[-1]:ax.set_xlabel('x (arcsec)')
    handles=[Line2D([],[],color='#252525',lw=.85,label='Truth'),Line2D([],[],color='#d55e00',lw=.75,label='Model'),
             Line2D([],[],marker='o',markersize=3,mfc='none',mec='#333333',ls='none',label='Images')]
    axes[0,2].legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,1.10),ncol=3,frameon=False,fontsize=7,handlelength=1.1,columnspacing=.6)
    fig.canvas.draw()
    for ax,mesh,label in zip(axes[0,:2],meshes,[r'$(\kappa_{\rm model}-\kappa_{\rm true})/\kappa_{\rm true}$',r'$(|\mu_{\rm model}|-|\mu_{\rm true}|)/|\mu_{\rm true}|$']):
        box=ax.get_position();cax=fig.add_axes([box.x0,box.y1+.025,box.width,.016])
        cb=fig.colorbar(mesh,cax=cax,orientation='horizontal',ticks=[-1,0,1,2],extend='both')
        cb.set_label(label,fontsize=8,labelpad=5);cb.ax.xaxis.set_label_position('top');cb.ax.xaxis.set_ticks_position('top')
        cb.ax.minorticks_off();cb.ax.tick_params(labelsize=8,length=2.5)
    fig.savefig(path,dpi=400,bbox_inches='tight',pad_inches=.04);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--render-only',action='store_true');a=p.parse_args()
    final=a.run/'final';cache=final/'tables/truth_recovery_grf_comparison.npz'
    cfg=json.loads((final/'artifacts/config.json').read_text());prefix=cfg['paths']['figure_prefix']
    output=final/'figures'/f'{prefix}_truth_recovery_residuals_grf_comparison.pdf'
    images=pd.read_csv(final/'tables/image_fit_quality.csv')
    if not a.render_only:
        config,state,evaluator,posterior,paths=load_run(a.run)
        if not (evaluator.gp_knee and state.gp_blocks=='all'):
            raise ValueError('A single cluster field requires the saved shared GRF model')
        # Exactly the parameter statistic used by compute_truth_recovery, not the best draw.
        theta=np.median(posterior.samples,axis=0)
        correction=evaluator.gp_deflection_correction(theta)
        if correction is None:raise ValueError('Conditional GRF not available for this saved model')
        z=float(config.truth.caustic_source_redshift)
        data={}
        for q in ['kappa','abs_mu','detA']:
            data[f'truth_{q}'],wcs=load_truth_fits(final/f'fits/truth_recovery_{q}_true.fits')
        shape=data['truth_kappa'].shape;x,y=grid_arcsec(state,wcs,shape)
        points=np.column_stack((x.ravel(),y.ravel()))
        packed=evaluator.packed_state_for_z(evaluator.physical(jnp.asarray(theta)),z)
        print(f'Exact parametric fields at {len(points)} map pixels',flush=True)
        _,u=evaluator.total_fields_at(points,packed,chunk=4096)
        ref=float(dpie_sigma0_factor_from_config(state.z_lens,state.gp_reference_redshift,state.cosmo_config))
        target=float(dpie_sigma0_factor_from_config(state.z_lens,z,state.cosmo_config))
        weight=target/ref/float(evaluator.gp_weights_per_family[0])
        du=[]
        for start in range(0,len(points),1024):
            block=points[start:start+1024];n=len(block);block=np.pad(block,((0,1024-n),(0,0)),mode='edge')
            f=correction(block[:,0],block[:,1],np.zeros(1024,dtype=int))
            du.append(np.stack(f[2:],-1)[:n].reshape(-1,2,2)*weight)
        du=np.concatenate(du)
        for name,jac in [('parametric',u),('with_grf',u+du)]:
            det=np.linalg.det(np.eye(2)-jac)
            data[f'{name}_kappa']=(.5*np.trace(jac,axis1=-2,axis2=-1)).reshape(shape)
            data[f'{name}_detA']=det.reshape(shape)
            with np.errstate(divide='ignore'):data[f'{name}_abs_mu']=np.abs(1/det).reshape(shape)
        differences={}
        for q in ['kappa','abs_mu','detA']:
            saved,_=load_truth_fits(final/f'fits/truth_recovery_{q}_model_median.fits')
            from astropy.io import fits
            header = fits.getheader(final/f'fits/truth_recovery_{q}_model_median.fits')
            mode = 'with_grf' if header.get('GRFMEAN', False) else 'parametric'
            np.testing.assert_allclose(data[f'{mode}_{q}'],saved,rtol=1e-8,atol=1e-10)
            differences[q]=float(np.nanmax(abs(data[f'{mode}_{q}']-saved)))
        data.update(x=x,y=y,grf_kappa=(.5*np.trace(du,axis1=-2,axis2=-1)).reshape(shape),theta=theta)
        np.savez_compressed(cache,**data)
        metadata=dict(run=str(a.run.resolve()),z_source=z,model='median latent parameter vector, matching the existing truth-recovery plot',
            grf='conditional mean at the same model and its reconstructed sources, not posterior-averaged fields',
            grid_shape=list(shape),redshift_weight=weight,baseline_max_absolute_difference=differences,
            magnification='1 / det(I - U_parametric - U_GRF), recomputed from the total Jacobian',
            normalization='display [-1,2], cropped from symmetric [-2,2] RdBu_r',
            grf_kappa_min=float(data['grf_kappa'].min()),grf_kappa_max=float(data['grf_kappa'].max()))
        (cache.with_suffix('.json')).write_text(json.dumps(metadata,indent=2)+'\n')
    with np.load(cache) as data:render(data,images,output)
    print(output,flush=True)


if __name__=='__main__':main()
