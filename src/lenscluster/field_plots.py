"""Conditional shared-field diagnostics at the highest-log-probability saved draw.

Maps retain the fitted projection and source positions. The displayed spectrum
is the homogeneous prior kernel before projection, not the map's Fourier power.
"""
from pathlib import Path
import json
import numpy as np


def render_grf_correction(data, output):
    """Standalone middle panel of the GRF overview, using the same saved map."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    from cmcrameri import cm
    from .plot_style import PAPER_STYLE
    x,y,grf,xy = (data[key] for key in ('x','y','grf','xy'))
    limit=max(.01,np.ceil(np.max(abs(grf))*100)/100)
    with plt.rc_context():
        plt.style.use(Path(__file__).with_name('style.mplstyle'))
        plt.rcParams.update(PAPER_STYLE)
        width=3.5
        map_width=2.8
        map_height=map_width*np.ptp(y)/np.ptp(x)
        height=map_height+1.05
        fig=plt.figure(figsize=(width,height))
        ax=fig.add_axes([.16,.48/height,map_width/width,map_height/height])
        cax=fig.add_axes([.16,(.48+map_height+.06)/height,map_width/width,.08/height])
        im=ax.pcolormesh(x,y,grf,cmap=cm.vik,
            norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),shading='nearest',rasterized=True)
        ax.scatter(*xy.T,s=data.get('image_sizes',9),facecolors='none',edgecolors='white',linewidths=.7,zorder=4)
        ax.scatter(*xy.T,s=data.get('image_sizes',9),facecolors='none',edgecolors='.12',linewidths=.3,zorder=5)
        if 'member_xy' in data:
            from matplotlib.lines import Line2D
            import matplotlib.patheffects as pe
            ax.scatter(*data['member_xy'].T,s=data.get('member_sizes',10),marker='D',facecolors='none',
                       edgecolors='#009E73',linewidths=.55,zorder=5)
            ax.scatter(*data['halo_xy'].T,s=48,marker='+',color='black',linewidths=1.,zorder=7,
                       path_effects=[pe.withStroke(linewidth=2.,foreground='white')])
            for (hx,hy),label in zip(data['halo_xy'],data['halo_labels']):
                if x[0]<=hx<=x[-1] and y[0]<=hy<=y[-1]:
                    offset=(4,5) if not label.startswith('BCG') else (4,-10)
                    ax.annotate(label,(hx,hy),xytext=offset,textcoords='offset points',fontsize=6,
                        zorder=8,path_effects=[pe.withStroke(linewidth=1.8,foreground='white')])
            handles=[Line2D([],[],marker='o',ms=3,mfc='none',mec='.12',ls='',label='Images (F200W)'),
                     Line2D([],[],marker='D',ms=3,mfc='none',mec='#009E73',ls='',label='Members (mass)'),
                     Line2D([],[],marker='+',ms=5,color='black',ls='',label='Independent halos')]
            ax.legend(handles=handles,loc='lower left',fontsize=6,frameon=True,
                      facecolor='white',edgecolor='none',framealpha=.85,handletextpad=.4)
        ax.set(xlim=(x[0],x[-1]),ylim=(y[0],y[-1]),aspect='equal',
               xlabel=r'$x$ (arcsec)',ylabel=r'$y$ (arcsec)')
        for axis in (ax.xaxis,ax.yaxis):
            axis.set_major_locator(MultipleLocator(40))
            axis.set_minor_locator(MultipleLocator(10))
        ax.tick_params(which='both',top=True,right=True)
        ax.grid(False)
        cb=fig.colorbar(im,cax=cax,orientation='horizontal')
        cb.set_label(r'$\delta\kappa_{\rm GRF}$',labelpad=5)
        cb.ax.xaxis.set_label_position('top')
        cb.ax.xaxis.set_ticks_position('top')
        cb.set_ticks([-.06,-.03,0,.03,.06])
        cb.ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        fig.savefig(output,dpi=400)
        plt.close(fig)


def render_grf(data, output):
    x,y,param,grf,total,xy,wave,power = [data[k] for k in
        ('x','y','parametric','grf','total','xy','wavelength_kpc','delta2_kappa')]
    turnover=float(data['turnover_kpc']); gamma=float(data['gamma']); OUT=output
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm,TwoSlopeNorm
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator,LogLocator,NullFormatter
    from lenscluster.plot_style import PAPER_STYLE
    from cmcrameri import cm
    plt.style.use(Path(__file__).with_name('style.mplstyle'));plt.rcParams.update(PAPER_STYLE)
    fig=plt.figure(figsize=(7.1,2.8));bottom=.20;width=.235;height=width*7.1/2.8*np.ptp(y)/np.ptp(x)
    positions=[.07,.325];positive=np.concatenate([param[param>0],total[total>0]]);norm=LogNorm(vmin=positive.min(),vmax=positive.max());lim=max(.01,np.ceil(np.max(abs(grf))*100)/100)
    for j,(values,cmap,n,label) in enumerate([(total,'inferno',norm,r'$\log_{10}\kappa_{\rm tot}$'),(grf,cm.vik,TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),r'$\delta\kappa_{\rm GRF}$')]):
     left=positions[j];ax=fig.add_axes([left,bottom,width,height]);cax=fig.add_axes([left,bottom+height+.025,width,.028])
     im=ax.pcolormesh(x,y,values,cmap=cmap,norm=n,shading='nearest',rasterized=True)
     ax.scatter(*xy.T,s=4,facecolors='none',edgecolors='white',linewidths=.6,zorder=4);ax.scatter(*xy.T,s=4,facecolors='none',edgecolors='.12',linewidths=.25,zorder=5)
     if j==0:
      for field,ls,lw in [(param,'--',.65),(total,'-',.8)]:
       cs=ax.contour(x,y,field,levels=[.4,.8,1.2],colors='white',linestyles=ls,linewidths=lw,alpha=.85)
       if ls=='-':ax.clabel(cs,fmt='%g',fontsize=6,inline=True,inline_spacing=3)
     if j==0:
      ax.legend(handles=[Line2D([],[],color='white',ls='--',lw=.65,label='Parametric'),Line2D([],[],color='white',lw=.8,label='+ GRF')],loc='upper left',fontsize=6.5,frameon=True,facecolor='black',edgecolor='none',framealpha=.55,labelcolor='white',borderpad=.3,handlelength=1.8,labelspacing=.25)
     ax.set(xlim=(x[0],x[-1]),ylim=(y[0],y[-1]),aspect='equal')
     ink='white' if j==0 else '.15'
     ax.set_xlabel(r'$x$ (arcsec)',fontsize=8)
     if j==0:ax.set_ylabel(r'$y$ (arcsec)',fontsize=8)
     else:ax.tick_params(labelleft=False)
     for axis in [ax.xaxis,ax.yaxis]:axis.set_major_locator(MultipleLocator(40));axis.set_minor_locator(MultipleLocator(10))
     ax.tick_params(which='both',top=True,right=True);ax.grid(False)
     cb=fig.colorbar(im,cax=cax,orientation='horizontal');cb.set_label(label,fontsize=9,labelpad=4);cb.ax.xaxis.set_label_position('top');cb.ax.xaxis.set_ticks_position('top');cb.ax.tick_params(labelsize=7,pad=2)
     if j==0:cb.set_ticks(10.**np.array([-.5,0,.5]));cb.set_ticklabels(['−0.5','0','0.5']);cb.minorticks_off()
     else:cb.set_ticks([-.06,-.03,0,.03,.06]);cb.ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    # Equal-height spectrum with inset labels, matching the map panels.
    ax=fig.add_axes([.655,bottom,.235,height]);order=np.argsort(wave);ax.loglog(wave[order],power[order],color='#0072B2',lw=1.4)
    ax.axvline(turnover,color='#D55E00',ls='--',lw=.8)
    ax.set(xlim=(wave.min(),wave.max()),xlabel='',ylabel='')
    ax.xaxis.label.set_size(8);ax.yaxis.label.set_size(9);ax.yaxis.tick_left();ax.yaxis.set_label_position('left');ax.yaxis.labelpad=3;ax.set_xticks([1,10,100]);ax.set_xticklabels(['1','10','100']);ax.yaxis.set_major_locator(LogLocator(base=10,numticks=5));ax.tick_params(which='both',top=True,right=True);ax.grid(False)
    ax.set_ylabel(r'$\Delta_\kappa^2(k)$',fontsize=9)
    ax.set_xlabel(r'$\lambda$ (kpc)',fontsize=8)
    ax.text(.95,.17,rf'$\lambda_{{\rm t}}={turnover:.0f}$ kpc'+'\n'+rf'$\gamma_\kappa={gamma:g}$',transform=ax.transAxes,ha='right',va='bottom',fontsize=8)
    fig.savefig(OUT,dpi=400);plt.close(fig)

def render_gp(a, output):
    x,y,xy=a['x'],a['y'],a['xy']
    d=a['at_images']*1000
    amplitude=np.linalg.norm(a['mean'],axis=-1)*1000
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MultipleLocator,MaxNLocator
    from lenscluster.plot_style import PAPER_STYLE
    plt.style.use(Path(__file__).with_name('style.mplstyle'));plt.rcParams.update(PAPER_STYLE)
    from scipy.spatial.distance import cdist
    from matplotlib.lines import Line2D
    colors=['#0072B2','#D55E00']
    # Conditional GP displacement and the fitted Gaussian correlation kernel.
    fig=plt.figure(figsize=(7.1,3.5));ax=fig.add_axes([.085,.17,.40,.40*7.1/3.5*np.ptp(y)/np.ptp(x)]);top=ax.get_position().y1;cbax=fig.add_axes([.085,top,.40,.027])
    rmax=max(1.,float(np.ceil(amplitude.max()/10)*10))
    m=ax.pcolormesh(x,y,amplitude,cmap='YlGnBu',vmin=0,vmax=rmax,shading='nearest',rasterized=True);ax.scatter(*xy.T,s=4,facecolors='none',edgecolors='.3',linewidths=.4)
    ax.set(xlim=(x[0],x[-1]),ylim=(y[0],y[-1]),aspect='equal',xlabel='$x$ (arcsec)',ylabel='$y$ (arcsec)');ax.tick_params(which='both',top=True,right=True);ax.grid(False)
    for axis in [ax.xaxis,ax.yaxis]:axis.set_major_locator(MultipleLocator(40));axis.set_minor_locator(MultipleLocator(10))
    
    cb=fig.colorbar(m,cax=cbax,orientation='horizontal');cb.ax.xaxis.set_ticks_position('top');cb.ax.xaxis.set_label_position('top');cb.set_label(r'GP correction $|\overline{\mathbf{d}}|$ (mas)',fontsize=8,labelpad=3);cb.locator=MaxNLocator(3);cb.update_ticks();cb.ax.tick_params(labelsize=7,direction='in')
    arrows=ax.quiver(*xy.T,*d.T,angles='xy',scale_units='xy',scale=10,color='.2',width=.003,headwidth=3.5,headlength=4,zorder=5)
    ax.quiverkey(arrows,.81,.92,50,'50 mas',labelpos='S',coordinates='axes',color='.2',labelcolor='.2',fontproperties={'size':7})
    ax2=fig.add_axes([.62,.17,.35,top-.17]);ell=float(a['ell']);rr=np.linspace(0,5*ell,501);rho=np.exp(-rr**2/(2*ell**2));ax2.plot(rr,rho,color=colors[0],lw=1.5)
    ax2.axvline(ell,color=colors[1],lw=.8,ls='--');ax2.plot(ell,np.exp(-.5),'o',color=colors[1],ms=3)
    dist=cdist(xy,xy);np.fill_diagonal(dist,np.inf);nearest=dist.min(1);shown=nearest[nearest<=5*ell];ax2.plot(shown,np.full(len(shown),.018),'|',color='.35',ms=5,mew=.65)
    ax2.set(xlim=(0,5*ell),ylim=(0,1.03),xlabel='$r$ (arcsec)',ylabel=r'$\rho(r)$');ax2.tick_params(which='both',top=True,right=True);ax2.grid(False)
    
    ax2.text(.97,.94,rf'$\ell_{{\rm GP}}={ell:.2f}^{{\prime\prime}}$'+'\n'+rf'$\sigma_{{\rm GP}}={1000*float(a["sigma"]):.1f}$ mas',transform=ax2.transAxes,va='top',ha='right',fontsize=8)
    ax2.legend(handles=[Line2D([],[],color=colors[0],lw=1.5,label='Fitted Gaussian kernel'),Line2D([],[],color='.35',marker='|',ls='',label='Nearest-image separation')],loc='center right',fontsize=6.5,frameon=False)
    out2=output;fig.savefig(out2,dpi=400);plt.close(fig)


def compute_fields(config, state, evaluator, posterior, *, step=.3, chunk=384):
    """Evaluate the fitted shared GRF and image GP without refitting or sampling."""
    import jax.numpy as jnp
    from scipy.linalg import cho_solve
    from scipy.integrate import simpson
    best = int(np.nanargmax(posterior.log_prob))
    theta = posterior.samples[best]
    np.testing.assert_allclose(evaluator.reported_physical(theta), posterior.physical[best], rtol=1e-8, atol=1e-8)
    e = evaluator._padded_evaluator()
    st = e.statistics(jnp.asarray(theta)); scales = e.error_scales(st['physical'])
    xy = np.column_stack([state.images.x_obs, state.images.y_obs]); n = len(xy)
    np.testing.assert_array_equal(np.asarray(e.gp['slots'])[0], np.arange(n))
    residual = np.asarray(st['fields']['backprojected'] - st['source_by_family'][np.asarray(state.images.family_index_per_image)])
    v = cho_solve((np.asarray(st['family_cholesky'])[0], True), residual.ravel()).reshape(n, 2)
    jacobian = np.asarray(st['fields']['jacobian'])
    u = np.einsum('nji,nj->ni', jacobian, v)
    sigma, ell = float(scales['sigma_shared']), float(scales['gp_ell'])
    def kernel(points):
        return sigma**2*np.exp(-np.sum((points[:,None]-xy[None])**2, axis=-1)/(2*ell**2))
    np.testing.assert_allclose(kernel(xy), np.asarray(e.gp_kernel_blocks(st['physical'], scales)['k_img'])[0], atol=1e-12)
    x = np.arange(xy[:,0].min()-8, xy[:,0].max()+8, step)
    y = np.arange(xy[:,1].min()-8, xy[:,1].max()+8, step)
    xx, yy = np.meshgrid(x, y); points = np.c_[xx.ravel(), yy.ravel()]
    correction = e.gp_deflection_correction(theta)
    weight = float(e.gp_weights_per_family[0])
    grf, mean = [], []
    for start in range(0, len(points), chunk):
        q = points[start:start+chunk]; count = len(q)
        padded = np.pad(q, ((0,chunk-count),(0,0)), mode='edge')
        fields = correction(padded[:,0], padded[:,1], np.zeros(chunk, dtype=int))
        grf.append(np.asarray((fields[2]+fields[5])/(2*weight))[:count])
        mean.append(kernel(q)@u)
    grf = np.concatenate(grf).reshape(xx.shape)
    zref = float(state.gp_reference_redshift)
    packed = e.packed_state_for_z(st['physical'], zref)
    _, hessian = e.total_fields_at(points, packed, chunk=2048)
    param = np.trace(hessian, axis1=1, axis2=2).reshape(xx.shape)/2
    knee, gamma, sig = (float(scales[k]) for k in ('gp_knee','gp_knee_gamma','gp_sigma_alpha'))
    lo, hi = config.likelihood.gp_knee_scales_arcsec
    k = np.geomspace(2*np.pi/hi, 2*np.pi/lo, 12001)
    shape = 1/(1+(k*knee/(2*np.pi))**gamma)
    power = shape*np.pi*sig**2/simpson(shape, x=np.log(k))
    np.testing.assert_allclose(simpson(power, x=np.log(k))/np.pi, sig**2, rtol=1e-12)
    data = dict(x=x,y=y,xy=xy,parametric=param,grf=grf,total=param+grf,
                mean=np.concatenate(mean).reshape(*xx.shape,2),at_images=kernel(xy)@u,
                sigma=sigma,ell=ell,draw_index=best,reference_redshift=zref,gamma=gamma,
                turnover_kpc=knee*state.kpc_per_arcsec,wavelength_kpc=2*np.pi/k*state.kpc_per_arcsec,
                delta2_kappa=k*k*power/(2*np.pi),k_arcsec_inverse=k,P_kappa_arcsec2=power)
    for key, value in data.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f'Nonfinite field diagnostic: {key}')
    return data


def write_field_plots(config, state, evaluator, posterior, paths, *, log=print):
    """Standard outputs for a single shared knee-spectrum GRF plus image GP."""
    import matplotlib.pyplot as plt
    if not getattr(evaluator, 'gp_knee', False):
        return []
    if config.likelihood.gp_blocks != 'all' or evaluator.gp_knee_spectrum != 'knee' or evaluator.gp_sigma_alpha_index < 0:
        log('[outputs] shared field maps skipped: require gp_blocks=all and an active knee-spectrum GRF')
        return []
    log('[outputs] computing shared GRF and GP diagnostic maps from the best saved draw')
    data = compute_fields(config, state, evaluator, posterior)
    paths.tables.mkdir(parents=True, exist_ok=True); paths.figures.mkdir(parents=True, exist_ok=True)
    prefix = f'{config.paths.figure_prefix}_' if config.paths.figure_prefix else ''
    np.savez_compressed(paths.tables / 'conditional_field_maps.npz', **data)
    outputs = [paths.figures / f'{prefix}total_grf_dimensionless_power.pdf',
               paths.figures / f'{prefix}gp_correction_correlation.pdf']
    with plt.rc_context():
        render_grf(data, outputs[0]); render_gp(data, outputs[1])
    provenance = dict(draw_index=int(data['draw_index']), reference_redshift=float(data['reference_redshift']),
                      conditioning='highest-log-probability saved draw; fitted source positions and projection retained',
                      spectrum='homogeneous kernel before projection; Delta_kappa^2=k^2 P_kappa/(2 pi)',
                      map='conditional mean, not posterior average or significance',
                      nonpositive_total_pixels=int(np.sum(data['total']<=0)),
                      nonpositive_display='masked in the logarithmic total-convergence panel')
    (paths.tables / 'conditional_field_maps.json').write_text(json.dumps(provenance, indent=2))
    for output in outputs:
        log(f'[outputs] {output}')
    return outputs
