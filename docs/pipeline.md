# Running a fit

{func}`lenscluster.pipeline.run` runs a fit in two stages, init and final. Run both, or one at a time:

```python
from lenscluster.pipeline import dry_run, run

dry_run(config)                   # describe the model, write nothing
result = run(config)              # init + final
init = run(config, stage="init")  # the search alone
result = run(config, stage="final")  # sample from a completed init
```

{func}`~lenscluster.pipeline.run_init` and {func}`~lenscluster.pipeline.run_final` are the two stages as separate
functions.

## Dry run

{func}`~lenscluster.pipeline.dry_run` builds the model and prints, as JSON, the number of parameters,
the provisional member classification, and the search and sampling sizes. It writes nothing.

## Init stage

The init stage finds the start of the final inference. It takes about a minute on Hera.

1. **Build** the model state from the configuration.
2. **Provisional classification** of the members into exact and far-field, with prior-inflated cut radii at the
   initial model.
3. **Search**, in two steps:
   - *Stage A*: the large-scale model (halos, shear, independent halos, sources), with the members on their relation,
     from `n_starts` prior draws. With `"lbfgs"`, every start first runs `screen_iterations`, and only the
     `screen_keep` best continue.
   - *Stage B*: from each carried mode, the members, with the large-scale parameters frozen, from `member_starts` prior
     draws of the member block. Besides the best modes, up to `diverse_carry` more distinct stage-A modes are carried,
     because a mode that is poor with the members on the relation can be the best once they deviate.

   The end points are clustered into modes, and the modes that can seed chains are polished in full dimension with
   L-BFGS.
4. **Final classification** at the best mode, and the classification report.
5. **Chain start**: every kept mode is optimized on the exact NUTS target, and the modes within `mode_margin` of the
   best seed the chains.

`SearchConfig.halo_method` and `member_method` choose the optimizer: scipy L-BFGS-B per start (`"lbfgs"`, default,
the fastest on CPU by a factor of four or more), vmapped optax L-BFGS (`"optax-lbfgs"`, for GPUs), Adam or mean-field
SVI.

### Verdict

The init log ends with a one-line verdict:

- **robust**: several independent starts reach the best basin;
- **fragile**: only one does. The result rests on a single optimization; consider more starts or a warm start.

### Search target and degenerate modes

The search maximizes the posterior density in the physical source coordinates. It leaves out the volume factor
$\log|\det L_\beta|$ of the source transport, which NUTS needs, because that factor grows without bound with the
magnification and gave a spurious mode at a lens so massive that every image was demagnified. As a safety net, a mode
whose median $|\det A|$ at the images exceeds `SearchConfig.max_median_abs_determinant` (10; cluster lenses sit near
0.2) is logged as degenerate and cannot seed chains.

### Warm start

```python
from dataclasses import replace

config = replace(config, search=replace(config.search, warm_start="results/demo/v1"),
                 paths=replace(config.paths, run_name="v2"))
```

The saved start of a previous run, matched by parameter name, replaces the search; the polish and chain-start step
still run. Parameters the old run lacks start at the configured model.

## Final stage

1. **NUTS** from the chain-start mode(s), with a dense mass matrix for the global block seeded by the Hessian, and the
   member offsets chosen by `dense_offset_members` in that block. Source and other member coordinates use a diagonal
   mass.
2. **Posterior audit** of the far-field approximation, including whether reclassifying at the posterior median would
   change the exact set. The result is in `final/artifacts/audit.json`.
3. **Outputs**: tables, figures and FITS maps under `final/` (see {doc}`outputs`).

## Resuming

`RuntimeConfig.resume` (on by default) reuses work in the run directory:

- A completed init is reused by any later final run whose model, likelihood, member and search configuration match.
- A completed final stage is reloaded when the whole configuration matches. A run with a different configuration under
  the same run name is refused rather than overwriting the saved posterior.

Use a new run label for each changed configuration.

## CPU chains

NUTS runs one chain per JAX CPU device. `RuntimeConfig(chains, nuts_chain_method, cpu_threads_per_chain)` sets the
plan, and {func}`lenscluster.cpu_runtime.configure_cpu_runtime` must run before JAX initializes:

```python
from lenscluster.cpu_runtime import configure_cpu_runtime

configure_cpu_runtime(4, "parallel")    # at the top of the script, before JAX starts
```

Building the configuration from a `.par` file starts JAX, so call it before that too. A late call raises an error
rather than silently running the chains on one device.

## Quick looks

For exploration, shorten the search and the chains and use independent GP blocks:

```python
from lenscluster.config import SamplingConfig

quick = replace(
    config,
    search=replace(config.search, n_starts=32),
    sampling=SamplingConfig(warmup=150, samples=100),
    likelihood=replace(config.likelihood, gp_blocks="system", gp_knee_project_modes=()),
    members=replace(config.members, far_field_cut_radii=2.0),
    paths=replace(config.paths, run_name="quick"),
)
```

A quick run checks that a model works; publish full runs. Check the audit's far-field error before trusting a run with
a looser far-field guard.
