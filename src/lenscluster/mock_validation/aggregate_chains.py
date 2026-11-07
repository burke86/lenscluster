"""Truth-centred mock chains, with recorded chain boundaries kept intact."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from ..block_corner import BLOCK_COLORS, STYLE, PAPER_STYLE
from .deviation_corners import load_columns, combined_draws
from .aggregate_plots import PARAMETER_LABELS


def chain_shape(path):
    """Verify the saved flattening order before reshaping physical draws."""
    with np.load(path, allow_pickle=False) as saved:
        grouped, flat, physical = saved['grouped_samples'], saved['samples'], saved['physical']
        if (grouped.ndim != 3 or flat.ndim != 2 or flat.shape != physical.shape
                or grouped.shape[-1] != flat.shape[-1] or grouped.size != flat.size):
            raise ValueError('Missing or inconsistent saved chain dimensions')
        if not np.array_equal(grouped.reshape(flat.shape), flat, equal_nan=True):
            raise ValueError('Grouped draws do not match flat draw order')
        return grouped.shape[:2]


def collect_chains(campaign):
    loaded, shapes, failures = {}, {}, {}
    for seed in campaign.seeds:
        try:
            shape = chain_shape(campaign.root/f'seed_{seed:08d}/solver/final/artifacts/posterior.npz')
            loaded[seed] = load_columns(campaign.root, seed)
            shapes[seed] = shape
        except (FileNotFoundError, KeyError, ValueError) as exc:
            failures[str(seed)] = str(exc)
    pooled = combined_draws(loaded)
    if pooled is None:
        return None, failures
    result = dict(parameters=pooled['parameters'], groups=pooled['groups'], mocks={})
    for seed in sorted(loaded):
        mask = pooled['seeds'] == seed
        if not mask.any():continue
        family_ids = list(dict.fromkeys(pooled['families'][mask].tolist()))
        nc, nd = shapes[seed]
        arrays = []
        for family in family_ids:
            take = mask & (pooled['families'] == family)
            np.testing.assert_array_equal(pooled['draw_indices'][take], np.arange(nc*nd))
            arrays.append(pooled['values'][take].reshape(nc, nd, -1))
        result['mocks'][seed] = dict(values=np.stack(arrays), families=family_ids)
    return result, failures


def convergence(chains):
    if chains.shape[0] < 2 or chains.shape[1] < 4:
        return np.nan, np.nan
    from numpyro.diagnostics import split_gelman_rubin, effective_sample_size
    return float(split_gelman_rubin(chains)), float(effective_sample_size(chains))


def write_aggregate_chains(campaign, output):
    data, failures = collect_chains(campaign)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    path = output/'mock_aggregate_chain_traces.pdf'
    if data is None:
        path.with_suffix('.json').write_text(json.dumps({'skipped': failures}, indent=2))
        return None
    names, groups, mocks = data['parameters'], data['groups'], data['mocks']
    seeds = list(mocks); nr, nc = len(names), len(seeds)
    styles = ['-', '--', ':', '-.']
    records = []
    with plt.rc_context(fname=STYLE, rc=PAPER_STYLE):
        fig, axes = plt.subplots(nr, nc, figsize=(max(7.1, 2.1*nc), 1.05*nr+.7), squeeze=False,
                                 sharex='col', sharey='row')
        for row, (name, group) in enumerate(zip(names, groups)):
            source = group == 'Sources'
            color = BLOCK_COLORS[group]
            field = name.split('.', 1)[1]
            symbol = {'beta_x':r'\beta_x', 'beta_y':r'\beta_y'}.get(field)
            if symbol is None:symbol=PARAMETER_LABELS.get(field, field).strip('$').replace(r'\log_{10}',r'\log')
            axes[row,0].set_ylabel(r'$\Delta('+symbol+r')/s_{68}$', color=color, fontsize=8)
            for col, seed in enumerate(seeds):
                ax = axes[row,col]
                values = mocks[seed]['values'][:,:,:,row]
                if not source:values=values[:1]
                rhats, esses = [], []
                for family, chains in enumerate(values):
                    for chain, trace in enumerate(chains):
                        ax.plot(np.arange(1,len(trace)+1), trace, color=color,
                                lw=.32 if source else .55, alpha=.15 if source else .65,
                                ls=styles[chain % len(styles)], rasterized=True)
                    rhat, ess = convergence(chains)
                    rhats.append(rhat);esses.append(ess)
                    records.append(dict(seed=seed, parameter=name,
                                        family=mocks[seed]['families'][family] if source else '',
                                        split_rhat=rhat, ess=ess))
                rhats=np.asarray(rhats);esses=np.asarray(esses)
                rhat = rhats[np.isfinite(rhats)].max() if np.isfinite(rhats).any() else np.nan
                ess = esses[np.isfinite(esses)].min() if np.isfinite(esses).any() else np.nan
                diagnostic=(rf'$\hat{{R}}={rhat:.3f}$' if np.isfinite(rhat) else r'$\hat{R}$ unavailable')
                if np.isfinite(ess):diagnostic+=f'  ESS {ess:.0f}'
                if source:diagnostic='Worst: '+diagnostic
                ax.text(.98,.96,diagnostic,transform=ax.transAxes,ha='right',va='top',fontsize=6,
                        bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=.4))
                if row==0:
                    ax.text(.02,.96,f'Mock {seed}',transform=ax.transAxes,va='top',fontsize=7)
                ax.axhline(0,color='.35',lw=.55,ls='--',zorder=0)
                ax.grid(False);ax.tick_params(labelsize=6,top=False,right=False)
                ax.yaxis.set_major_locator(MaxNLocator(3))
                ax.xaxis.set_major_locator(MaxNLocator(3,integer=True))
                if row==0 or groups[row]!=groups[row-1]:
                    ax.spines['top'].set_color(color);ax.spines['top'].set_linewidth(1.5)
                if row==nr-1:ax.set_xlabel('Post-warmup draw',fontsize=8)
        handles=[Line2D([],[],color=c,lw=2,label=g) for g,c in BLOCK_COLORS.items() if g in groups]
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.999),ncol=5,frameon=False,fontsize=11)
        fig.subplots_adjust(left=.065,right=.995,bottom=.025,top=.98,hspace=.06,wspace=.04)
        fig.savefig(path,bbox_inches='tight',pad_inches=.06,dpi=160)
        plt.close(fig)
    pd.DataFrame(records).to_csv(path.with_suffix('.csv'),index=False)
    path.with_suffix('.json').write_text(json.dumps(dict(
        parameters=names.tolist(),groups=groups.tolist(),colors=BLOCK_COLORS,
        normalization='(p-truth)/((q84-q16)/2), per mock and per source coordinate; identical to combined truth-deviation corner',
        chains='Each mock in its own column; chains never joined; line styles cycle solid/dashed/dotted/dash-dot',
        sources='All family x/y chains overlaid individually, not averaged. Worst annotation uses maximum finite Rhat and minimum finite ESS across families.',
        diagnostics='NumPyro split Gelman-Rubin and effective sample size, per mock and physical parameter; never pooled across mocks. Undefined diagnostics omitted from worst-finite summary.',
        mocks={str(seed):dict(shape=list(m['values'].shape),families=m['families']) for seed,m in mocks.items()},
        skipped=failures),indent=2))
    return path
