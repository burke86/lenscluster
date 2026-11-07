"""GP diagnostics using only the current run, without fold-sample panels."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

GP_COLOR, WHITE_COLOR = '#25758a', '#c27645'
INK = '#303943'


def apply_style():
    """Use the repository's publication stylesheet without visual overrides."""
    plt.style.use('default')
    plt.style.use(str(Path(__file__).with_name('style.mplstyle')))


def save(fig, output):
    for ax in fig.axes:
        if hasattr(ax, '_colorbar') or not ax.axison:
            continue
        if ax.get_aspect() == 1:
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            half_span = max(abs(xlim[1] - xlim[0]), abs(ylim[1] - ylim[0])) / 2
            for limits, setter in ((xlim, ax.set_xlim), (ylim, ax.set_ylim)):
                center = np.mean(limits)
                direction = np.sign(limits[1] - limits[0])
                setter(center - direction * half_span, center + direction * half_span)
        ax.set_box_aspect(1)
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.tick_params(which='both', top=True, right=True)
    fig.canvas.draw()
    for ax in fig.axes:
        if hasattr(ax, '_colorbar') or not ax.axison:
            continue
        box = ax.get_window_extent()
        assert np.isclose(box.width, box.height, rtol=1e-6), 'Plot axes must be square'
    if isinstance(output, PdfPages):
        output.savefig(fig)
    else:
        fig.savefig(output)
    plt.close(fig)
    return output


def plot_kernel_sky(state, evaluator, posterior, output):
    xy=np.column_stack([state.images.x_obs,state.images.y_obs])
    slots,mask=(np.asarray(evaluator.gp[k]) for k in ('slots','mask'))
    names=[p.name for p in state.parameter_specs]
    shared=float(np.median(posterior.physical[:,names.index('images.sigma_shared')]))
    white=float(np.median(posterior.physical[:,names.index('images.sigma_white')]))
    inst=np.asarray(state.images.sigma_inst_per_image)
    variance=shared**2+white**2+inst**2
    ell=(float(np.median(posterior.physical[:,names.index('images.gp_ell')]))
         if 'images.gp_ell' in names else state.gp_correlation_length_arcsec)
    point_correlations=np.zeros(len(xy))
    segments=[]; correlations=[];separations=[]
    for f in range(len(slots)):
        rows=slots[f][mask[f]]
        for n,i in enumerate(rows):
            for j in rows[n+1:]:
                sep=np.linalg.norm(xy[i]-xy[j])
                rho=shared**2*np.exp(-sep**2/(2*ell**2))/np.sqrt(variance[i]*variance[j])
                segments.append([xy[i],xy[j]]);correlations.append(rho);separations.append(sep)
                point_correlations[i]=max(point_correlations[i],rho)
                point_correlations[j]=max(point_correlations[j],rho)
    correlations=np.asarray(correlations);separations=np.asarray(separations)
    fig,(sky,curve)=plt.subplots(1,2,figsize=(14,7))
    visible=correlations>=.2
    norm=plt.Normalize(0,1)
    lines=LineCollection(np.asarray(segments)[visible],cmap='viridis',norm=norm,linewidths=2.2)
    lines.set_array(correlations[visible])
    sky.add_collection(lines)
    sky.scatter(*xy.T,s=32,c=point_correlations,cmap='viridis',norm=norm,zorder=3,edgecolors='white',linewidths=.4)
    cbar=fig.colorbar(lines,ax=sky,fraction=.035,pad=.035)
    cbar.set_label(r'Total image-plane correlation $\rho_{ij}$')
    sky.set_aspect('equal');sky.invert_xaxis()
    sky.set_xlabel('x (arcsec)');sky.set_ylabel('y (arcsec)')
    d=np.geomspace(.1,max(separations.max()*1.05,30),500)
    k=np.exp(-d**2/(2*ell**2))
    curve.plot(d,k,color='.5',ls='--',lw=2,label='Shared component alone')
    curve.plot(d,shared**2*k/variance.mean(),color=GP_COLOR,lw=2.4,label='Including white + instrumental noise')
    curve.scatter(separations,correlations,s=24,color=GP_COLOR,alpha=.4,linewidths=0)
    curve.set(xscale='log',xlim=(.1,d.max()),ylim=(-.03,1.16),
              xlabel='Same-family image separation (arcsec)',ylabel='Image-plane displacement correlation')
    curve.legend(loc='lower left',bbox_to_anchor=(.015,.12),fontsize=12)
    curve.text(.04,.95,rf'Correlation length $\ell={ell:.2g}$ arcsec'+(' (median)' if 'images.gp_ell' in names else ' (fixed)'),transform=curve.transAxes,va='top')
    fig.text(.5,.025,r'Colors: pair correlation on links; strongest same-family correlation at each point. Links require $\rho\geq0.2$.',ha='center',fontsize=12)
    fig.tight_layout(rect=(0,.06,1,1),w_pad=3)
    return save(fig,output)


def pick_family(state, evaluator, min_images=3, closest=1.5, widest=10.0):
    """A family with at least ``min_images`` images, none closer than ``closest`` arcsec and none farther than ``widest``."""
    xy = np.column_stack([state.images.x_obs, state.images.y_obs])
    slots, mask = np.asarray(evaluator.gp['slots']), np.asarray(evaluator.gp['mask'])
    best = None
    for f in range(slots.shape[0]):
        rows = slots[f][mask[f]]
        if len(rows) < min_images:
            continue
        seps = [np.linalg.norm(xy[i] - xy[j]) for a, i in enumerate(rows) for j in rows[a + 1:]]
        if min(seps) >= closest and max(seps) <= widest and (best is None or len(rows) > len(best[1])):
            best = f, rows
    if best is None:
        f = int(np.argmax(mask.sum(1)))
        best = f, slots[f][mask[f]]
    return best


def plot_shared_vs_white(state, evaluator, posterior, output):
    names=[p.name for p in state.parameter_specs]
    shared=float(np.median(posterior.physical[:,names.index('images.sigma_shared')]))
    white=float(np.median(posterior.physical[:,names.index('images.sigma_white')]))
    ell=(float(np.median(posterior.physical[:,names.index('images.gp_ell')]))
         if 'images.gp_ell' in names else state.gp_correlation_length_arcsec)
    inst=np.asarray(state.images.sigma_inst_per_image)
    labels=list(state.images.image_labels)
    if '1.b' in labels and '1.c' in labels:
        i,j=labels.index('1.b'),labels.index('1.c')
    else:
        _,rows=pick_family(state,evaluator)
        i,j=rows[:2]
    xy=np.column_stack([state.images.x_obs,state.images.y_obs])
    sep=np.linalg.norm(xy[i]-xy[j]);inst_pair=np.sqrt((inst[i]**2+inst[j]**2)/2)
    d=np.linspace(0,25,500);k=np.exp(-d**2/(2*ell**2))
    terms=[shared*np.sqrt(2*(1-k)),np.full_like(d,np.sqrt(2)*white),np.full_like(d,np.sqrt(2)*inst_pair)]
    fig,axes=plt.subplots(1,2,figsize=(14,7))
    ax=axes[0]
    for values,color,style,label in zip(terms,[GP_COLOR,WHITE_COLOR,'.5'],['--',':','-.'],['Shared contribution','White contribution','Instrumental contribution']):
        ax.plot(d,1e3*values,color=color,ls=style,lw=2,label=label)
    ax.plot(d,1e3*np.sqrt(sum(v*v for v in terms)),color=INK,lw=2.4,label='Combined in quadrature')
    ax.set(xlim=(0,25),ylim=(0,300),xlabel='Image-pair separation (arcsec)',
           ylabel=r'Uncertainty in relative displacement $d_i-d_j$ (mas)')
    ax.legend(loc='upper left',fontsize=12)
    ax=axes[1]
    va=shared**2+white**2+inst[i]**2;vb=shared**2+white**2+inst[j]**2
    rho=shared**2*np.exp(-sep**2/(2*ell**2))/np.sqrt(va*vb)
    for probability,ls in [(.954499736,'--'),(.682689492,'-')]:
        radius=np.sqrt(-2*np.log(1-probability))
        ax.add_patch(Ellipse((0,0),2*radius,2*radius,facecolor='none',edgecolor='.5',ls=ls,lw=1.6))
        ax.add_patch(Ellipse((0,0),2*radius*np.sqrt(1+rho),2*radius*np.sqrt(1-rho),angle=45,
                            facecolor=GP_COLOR,edgecolor='none',alpha=.10))
        ax.add_patch(Ellipse((0,0),2*radius*np.sqrt(1+rho),2*radius*np.sqrt(1-rho),angle=45,
                            facecolor='none',edgecolor=GP_COLOR,ls=ls,lw=2))
    ax.set(xlim=(-3.8,3.8),ylim=(-3.8,3.8),
           xlabel=rf'Displacement of {labels[i]} / $\sigma_i$',ylabel=rf'Displacement of {labels[j]} / $\sigma_j$')
    ax.text(.04,.96,f'{labels[i]} / {labels[j]}: {sep:.2f} arcsec apart',transform=ax.transAxes,va='top')
    ax.text(.04,.04,rf'$\rho={rho:.3f}$'+'\nTilted ellipse: images move together',transform=ax.transAxes,fontsize=12)
    fig.text(.5,.055,'Right: GP in blue; independent errors with the same marginal variance in gray. Solid/dashed contours enclose 68.3%/95.4%.',ha='center',fontsize=12)
    fig.text(.5,.025,'These are conditional noise distributions at median fitted GP parameters, not measured image displacements.',ha='center',fontsize=12)
    fig.tight_layout(rect=(0,.08,1,1),w_pad=3)
    return save(fig,output)


DEFAULT_SCALE_SITES = (('images.sigma_shared', 'Shared displacement scale', r'$\sigma_{\rm shared}$', 1e3, 'mas'),
                       ('images.sigma_white', 'Independent displacement scale', r'$\sigma_{\rm white}$', 1e3, 'mas'))
KNEE_SCALE_SITES = (('images.gp_sigma_alpha', 'Mass-field deflection error', r'$\sigma_\alpha$', 1e3, 'mas'),
                    ('images.gp_knee', 'Knee wavelength', r'$\ell_{\rm knee}$', 1.0, 'arcsec'),
                    ('images.gp_knee_gamma', 'Spectral slope beyond the knee', r'$\gamma_{\rm GRF}$', 1.0, ''),
                    ('images.sigma_shared', 'Image-plane GP amplitude', r'$\sigma_{\rm shared}$', 1e3, 'mas'),
                    ('images.gp_ell', 'Image-plane GP length', r'$\ell$', 1.0, 'arcsec'),
                    ('images.sigma_white', 'Independent position scatter', r'$\sigma_{\rm white}$', 1e3, 'mas'))


def plot_scale_posteriors(state, posterior, output, sites=DEFAULT_SCALE_SITES):
    """Posterior against prior for each sampled image-error site present in the state; ``sites`` lists (name, description,
    symbol, display factor, unit)."""
    from lenscluster.corner_plots import DisplayPrior
    names=[p.name for p in state.parameter_specs]
    sites=[site for site in sites if site[0] in names]
    fig,axes=plt.subplots(1,len(sites),figsize=(6.5*len(sites),6.8))
    axes=np.atleast_1d(axes)
    for ax,(name,description,symbol,factor,unit) in zip(axes,sites):
        i=names.index(name);spec=state.parameter_specs[i]
        values=np.asarray(posterior.physical[:,i]);q16,q50,q84=np.percentile(values,[16,50,84])
        prior=DisplayPrior(spec)
        maximum=max(np.percentile(values,99.9)*1.4,q84*1.8)
        x=np.linspace(max(values.min()/20,1e-6),maximum,600)
        pdf=prior.pdf(np.log10(x))/(x*np.log(10)) if spec.transform_kind=='log_positive' else prior.pdf(x)
        ax.hist(values*factor,bins=30,density=True,color=GP_COLOR,alpha=.35,histtype='stepfilled',label='Posterior')
        ax.hist(values*factor,bins=30,density=True,color=GP_COLOR,histtype='step',lw=1.6)
        ax.plot(x*factor,pdf/factor,color=WHITE_COLOR,ls='--',lw=2,label='Prior')
        ax.axvspan(q16*factor,q84*factor,color=GP_COLOR,alpha=.10,zorder=-3)
        ax.axvline(q50*factor,color=GP_COLOR,lw=1.4,ls=':')
        ax.set_xlim(0,maximum*factor);ax.set_ylim(0,ax.get_ylim()[1]*1.32)
        unit_label=f' ({unit})' if unit else ''
        ax.set_xlabel(description+'\n'+symbol+unit_label);ax.set_ylabel('Probability density'+(f' ({unit}$^{{-1}}$)' if unit else ''))
        digits=0 if factor>1 else 3
        ax.text(.96,.95,rf'${q50*factor:.{digits}f}^{{+{(q84-q50)*factor:.{digits}f}}}_{{-{(q50-q16)*factor:.{digits}f}}}$ {unit}',transform=ax.transAxes,ha='right',va='top')
        ax.legend(loc='center right')
    fig.text(.5,.055,'Shading marks the central 68% posterior interval; dotted lines mark the medians.',ha='center',fontsize=12)
    fig.text(.5,.025,'Display ranges focus on the fitted scales. Prior curves keep their full normalization.',ha='center',fontsize=12)
    fig.tight_layout(rect=(0,.08,1,1),w_pad=3)
    return save(fig,output)


def plot_knee_kernel(state, evaluator, posterior, output):
    """Longitudinal and transverse correlations of the fitted mass field (median knee wavelength) and the image-plane GP
    kernel, against separation, with the same-image-pair separations of the catalog marked."""
    from lenscluster.gp_knee import knee_curves
    names=[p.name for p in state.parameter_specs]
    med=lambda n: float(np.median(posterior.physical[:,names.index(n)])) if n in names else None
    lam, ell, slope = med('images.gp_knee'), med('images.gp_ell'), med('images.gp_knee_gamma')
    tables=evaluator.gp_tables; r=np.asarray(tables['r_grid'])
    fig,ax=plt.subplots(figsize=(7,7))
    if lam is not None:
        cl,ct=(np.asarray(v) for v in knee_curves(tables, lam, slope))
        slope_label = '' if slope is None else f', $p={slope:.2g}$'
        ax.plot(r,ct,color=GP_COLOR,lw=2.4,label=rf'mass field, transverse ($\ell_{{\rm knee}}={lam:.2g}$"{slope_label})')
        ax.plot(r,cl,color=WHITE_COLOR,lw=2.4,label='mass field, longitudinal')
    ax.plot(r,np.exp(-r**2/(2*ell**2)),color=INK,lw=1.8,ls='--',label=rf'image-plane GP ($\ell={ell:.2g}$")')
    ax.axhline(0,color='.5',lw=1)
    xy=np.column_stack([state.images.x_obs,state.images.y_obs]); fam=np.asarray(state.images.family_index_per_image)
    i,j=np.triu_indices(len(xy),1); sep=np.hypot(*(xy[i]-xy[j]).T)
    ax.plot(sep[fam[i]==fam[j]],np.full((fam[i]==fam[j]).sum(),-0.45),'|',color=INK,alpha=.5,ms=8,label='same-family image pairs')
    ax.set(xscale='log',xlim=(.1,max(r.max(),30)),ylim=(-.55,1.1),xlabel='Separation (arcsec)',ylabel='Correlation')
    ax.legend(loc='lower left',fontsize=11)
    fig.tight_layout()
    return save(fig,output)


def write_gp_plots(config, state, evaluator, posterior, paths, *, log=print):
    """Write the three self-contained diagnostics for GP runs only."""
    if not getattr(evaluator, "gp_mode", False):
        return []
    paths.figures.mkdir(parents=True, exist_ok=True)
    prefix = f"{config.paths.figure_prefix}_" if config.paths.figure_prefix else ""
    with plt.rc_context():
        apply_style()
        if getattr(evaluator, "gp_knee", False):
            outputs = [
                plot_knee_kernel(state, evaluator, posterior, paths.figures / f"{prefix}gp_kernel_curves.pdf"),
                plot_scale_posteriors(state, posterior, paths.figures / f"{prefix}gp_scale_posteriors.pdf", KNEE_SCALE_SITES),
            ]
        else:
            outputs = [
                plot_kernel_sky(state, evaluator, posterior, paths.figures / f"{prefix}gp_kernel_sky.pdf"),
                plot_shared_vs_white(state, evaluator, posterior, paths.figures / f"{prefix}gp_shared_vs_white.pdf"),
                plot_scale_posteriors(state, posterior, paths.figures / f"{prefix}gp_scale_posteriors.pdf"),
            ]
    from .field_plots import write_field_plots
    outputs.extend(write_field_plots(config, state, evaluator, posterior, paths, log=log))
    for output in outputs:
        log(f"[outputs] {output}")
    return outputs
