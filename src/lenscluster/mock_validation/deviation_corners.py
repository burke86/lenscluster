"""Within-component corners of truth-centred joint posterior draws across mocks."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .aggregate_plots import STYLE, PAPER_STYLE, PAPER_PARAMETERS, PARAMETER_LABELS, _save
from ..utils import log10_dpie_mass_msun


def standardized_draws(values, truth):
    """Truth-centred draws in half-central-68%-interval units; no mean removal."""
    values=np.asarray(values,float);truth=np.asarray(truth,float)
    lo,hi=np.percentile(values,[16,84],axis=0)
    scale=(hi-lo)/2
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(truth)) or np.any(scale<=0):
        raise ValueError('Nonfinite or constant posterior coordinate')
    return (values-truth)/scale,scale


def load_columns(root,seed):
    folder=root/f'seed_{seed:08d}'
    state=json.loads((folder/'solver/init/artifacts/state.json').read_text())
    with np.load(folder/'solver/final/artifacts/posterior.npz',allow_pickle=False) as saved:
        physical=saved['physical']
    specs=state['parameter_specs']
    if physical.shape[1]!=len(specs):raise ValueError('Saved parameter count mismatch')
    columns={s['name']:physical[:,i] for i,s in enumerate(specs)}
    for prefix,v,r in [('halo','v_disp','cut_radius_kpc'),('bcg','v_disp','cut_radius_kpc'),('potfile','sigma','cutkpc')]:
        if prefix+'.'+v in columns and prefix+'.'+r in columns:
            columns[prefix+'.log10_mass_msun']=log10_dpie_mass_msun(columns[prefix+'.'+v],columns[prefix+'.'+r])
    if 'potfile.alpha_sigma' in columns and 'potfile.beta_radius' in columns:
        columns['potfile.gamma_ml']=2*columns['potfile.alpha_sigma']+columns['potfile.beta_radius']-1
    recovery=json.loads((folder/'recovery/recovery.json').read_text())
    truth={r['parameter']:r['truth'] for r in recovery['parameters']}
    return columns,truth


def write_deviation_corners(campaign,output):
    import corner
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    loaded={};failures={}
    for seed in campaign.seeds:
        try:loaded[seed]=load_columns(campaign.root,seed)
        except (FileNotFoundError,KeyError,ValueError) as exc:failures[str(seed)]=str(exc)
    definitions=[('halo','halo.','tab:blue'),('bcg','bcg.','tab:purple'),
                 ('grf_gp','images.','tab:red'),('scaling_relation','potfile.','tab:orange'),
                 ('sources','source.','tab:green')]
    outputs=[];metadata=dict(definition='u=(p-p_true)/s68; s68=(q84-q16)/2 from each mock posterior',
                            weighting='Each mock equal; sources equal within each mock. All finite saved joint draws.',
                            interpretation='Mixture across truth-centred mocks, not a single-cluster posterior; correlations include between-mock bias.',
                            skipped_parameters={'halo.log10_aperture_mass_msun':'No joint aperture-mass draws saved'},
                            load_failures=failures,blocks={})
    for block,prefix,color in definitions:
        if block=='sources':names=['source.beta_x','source.beta_y']
        else:
            names=[p for p in PAPER_PARAMETERS if p.startswith(prefix) and not p.endswith('log10_aperture_mass_msun')]
            names=[p for p in names if loaded and all(p in c and np.isfinite(t.get(p,np.nan)) and np.isfinite(c[p]).all() and np.ptp(c[p])>0 for c,t in loaded.values())]
        if not names:continue
        arrays=[];weights=[];seeds=[];scales_saved={}
        for seed,(columns,truth) in loaded.items():
            local=[];scales={}
            pairs=([names] if block!='sources' else
                   [[name,name[:-1]+'y'] for name in columns if name.startswith('source.') and name.endswith('.beta_x') and name[:-1]+'y' in columns])
            for pair in pairs:
                if not all(p in truth and np.isfinite(truth[p]) for p in pair):continue
                try:values,scale=standardized_draws(np.column_stack([columns[p] for p in pair]),[truth[p] for p in pair])
                except ValueError:continue
                local.append(values);scales['|'.join(pair)]=scale.tolist()
            if not local:continue
            values=np.concatenate(local);arrays.append(values)
            weights.append(np.full(len(values),1/len(values)));seeds.append(np.full(len(values),seed))
            scales_saved[str(seed)]=scales
        if not arrays:continue
        values=np.concatenate(arrays);weight=np.concatenate(weights);weight/=weight.sum()
        n=len(names)
        labels=[]
        for name in names:
            field=name.split('.',1)[1]
            symbol={'beta_x':r'\beta_x','beta_y':r'\beta_y'}.get(field)
            if symbol is None:symbol=PARAMETER_LABELS.get(field,field).strip('$').replace(r'\log_{10}',r'\log')
            labels.append(r'$\Delta('+symbol+r')/s_{68}$')
        # Full ranges preserve outliers rather than silently trimming the mocks.
        ranges=[(min(0.,values[:,i].min())-.04*np.ptp(values[:,i]),max(0.,values[:,i].max())+.04*np.ptp(values[:,i])) for i in range(n)]
        with plt.rc_context(fname=STYLE,rc=PAPER_STYLE):
            fig=plt.figure(figsize=(max(4,1.65*n),max(4,1.65*n)))
            corner.corner(values,weights=weight,labels=labels,range=ranges,bins=40,smooth=1.,smooth1d=1.,
                          color=color,levels=(.68,.95),plot_datapoints=False,plot_density=False,
                          fill_contours=True,show_titles=False,max_n_ticks=3,fig=fig,
                          hist_kwargs={'linewidth':2.0},contour_kwargs={'linewidths':1.8},label_kwargs={'fontsize':9})
            axes=np.asarray(fig.axes).reshape(n,n)
            from ..block_corner import annotate_corner_summary
            for i in range(n):
                annotate_corner_summary(axes[i,i], values[:,i], weight, color=color)
                axes[i,i].axvline(0,color='.25',ls='--',lw=1.0)
                for j in range(i):
                    axes[i,j].axvline(0,color='.3',ls='--',lw=1.0)
                    axes[i,j].axhline(0,color='.3',ls='--',lw=1.0)
                    axes[i,j].plot(0,0,'x',color='.15',ms=4,mew=1)
                for j in range(n):
                    axes[i,j].grid(False);axes[i,j].tick_params(labelsize=7)
            fig.subplots_adjust(hspace=.04,wspace=.04)
            path=output/f'mock_{block}_truth_deviation_corner.pdf';_save(fig,path);outputs.append(path)
        np.savez_compressed(output/f'mock_{block}_truth_deviation_corner.npz',values=values,weights=weight,seeds=np.concatenate(seeds),parameters=np.asarray(names))
        metadata['blocks'][block]=dict(parameters=names,scales=scales_saved,draw_rows=len(values),seeds=sorted(scales_saved))
    (output/'mock_truth_deviation_corners.json').write_text(json.dumps(metadata,indent=2))
    return outputs


def combined_draws(loaded):
    """Pool aligned global/source draws, with equal mock and family weights."""
    blocks = [('Halo', 'halo.', 'tab:blue'), ('BCG', 'bcg.', 'tab:purple'),
              ('GRF + GP', 'images.', 'tab:red'),
              ('Scaling relation', 'potfile.', 'tab:orange')]
    names, colors, groups = [], [], []
    for label, prefix, color in blocks:
        selected = [p for p in PAPER_PARAMETERS if p.startswith(prefix)
                    and not p.endswith('log10_aperture_mass_msun')
                    and all(p in c and np.isfinite(t.get(p, np.nan))
                            and np.isfinite(c[p]).all() and np.ptp(c[p]) > 0
                            for c, t in loaded.values())]
        names.extend(selected); colors.extend([color] * len(selected))
        groups.extend([label] * len(selected))
    global_names = list(names)
    has_sources = bool(loaded) and all(any(
        p.startswith('source.') and p.endswith('.beta_x') and p[:-1]+'y' in c
        for p in c) for c, _ in loaded.values())
    if has_sources:
        names.extend(['source.beta_x', 'source.beta_y'])
        colors.extend(['tab:green'] * 2); groups.extend(['Sources'] * 2)
    arrays, weights, seeds, families, draw_indices = [], [], [], [], []
    for seed, (columns, truth) in loaded.items():
        pairs = [[p, p[:-1]+'y'] for p in sorted(columns)
                 if p.startswith('source.') and p.endswith('.beta_x')
                 and p[:-1]+'y' in columns] if has_sources else [[]]
        local = []
        for pair in pairs:
            parameters = global_names + pair
            if not parameters or not all(np.isfinite(truth.get(p, np.nan)) for p in parameters):
                continue
            try:
                values, _ = standardized_draws(
                    np.column_stack([columns[p] for p in parameters]),
                    [truth[p] for p in parameters])
            except ValueError:
                continue
            local.append((values, pair[0].split('.')[1] if pair else ''))
        for values, family in local:
            arrays.append(values)
            weights.append(np.full(len(values), 1 / (len(local) * len(values))))
            seeds.extend([seed] * len(values)); families.extend([family] * len(values))
            draw_indices.extend(range(len(values)))
    if not arrays:
        return None
    weights = np.concatenate(weights); weights /= weights.sum()
    return dict(values=np.concatenate(arrays), weights=weights, seeds=np.asarray(seeds),
                families=np.asarray(families), draw_indices=np.asarray(draw_indices),
                parameters=np.asarray(names), colors=np.asarray(colors), groups=np.asarray(groups))


def write_combined_deviation_corner(campaign, output):
    """One corner with row-block colors and intact cross-block joint correlations."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    loaded, failures = {}, {}
    for seed in campaign.seeds:
        try:
            loaded[seed] = load_columns(campaign.root, seed)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            failures[str(seed)] = str(exc)
    data = combined_draws(loaded)
    if data is None:
        return None
    values, weights = data['values'], data['weights']
    names, groups = data['parameters'], data['groups']
    labels = []
    for name in names:
        field = name.split('.', 1)[1]
        symbol = {'beta_x': r'\beta_x', 'beta_y': r'\beta_y'}.get(field)
        if symbol is None:
            symbol = PARAMETER_LABELS.get(field, field).strip('$').replace(r'\log_{10}', r'\log')
        labels.append(r'$\Delta(' + symbol + r')/s_{68}$')
    from ..block_corner import render_block_corner
    path = output/'mock_combined_truth_deviation_corner.pdf'
    render_block_corner(values, weights, labels, groups, path, zero_reference=True, summarize=True)
    np.savez_compressed(path.with_suffix('.npz'), **data)
    path.with_suffix('.json').write_text(json.dumps(dict(
        definition='(parameter-truth)/half posterior 16-84 interval, separately for each mock',
        weighting='Equal mock weight; equal family weight within mock; all saved joint draws',
        sources='Global rows repeated for each paired source x/y, preserving original draw alignment',
        interpretation='Pooled truth-centred posterior mixture, not a single-cluster posterior',
        contour_color='Row block', load_failures=failures,
        seeds=sorted(set(data['seeds'].tolist())), rows=len(values)), indent=2))
    return path
