#!/usr/bin/env python
"""User-launched feasibility workflow. UNTESTED: written without execution."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
# Must precede imports of JAX or the model adapter.
os.environ.setdefault('JAX_ENABLE_X64','true')
os.environ.setdefault('MPLCONFIGDIR','/tmp/lenscluster-mpl')
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/lenscluster-numba')
OUT=ROOT/'feasibility'

STEPS={
 'catalogs':('catalogs','audit',()),
 'checks':('validation','run_checks',('catalogs',)),
 'optimize':('fitting','optimize',('checks',)),
 'laplace':('fitting','laplace',('optimize',)),
 'nuts':('fitting','nuts',('optimize',)),
 'compare':('fitting','compare',('laplace','nuts')),
 'residuals':('fitting','residuals',('optimize',)),
 'benchmark':('benchmarks','benchmark',('nuts',)),
 'ratios-optimize':('fitting','optimize',('optimize',)),
 'ratios-nuts':('fitting','nuts',('ratios-optimize',)),
 'fisher':('cosmography','fisher',('optimize',)),
 'maps':('cosmography','maps',('nuts',)),
 'area':('area','areas',('nuts',)),
 'plots':('reporting','plots',()),
 'report':('reporting','report',()),
}

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def archive_outputs():
    """Explicit --restart only: preserve generated files, never source/docs."""
    import uuid
    suffixes={'.json','.npz','.csv','.cat','.pdf','.log','.fits','.png'}
    paths=[p for p in OUT.iterdir() if p.is_file() and
           (p.suffix in suffixes or p.name in ('REPORT.md','summary.md','inventory.md'))]
    for directory in ('cosmography','failures'):
        paths.extend(p for p in (OUT/directory).rglob('*') if p.is_file() and
                     (p.suffix in suffixes or (directory=='failures' and p.suffix=='.txt')))
    if not paths:return None
    archive=OUT/'archives'/(time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8])
    for path in paths:
        target=archive/path.relative_to(OUT);target.parent.mkdir(parents=True,exist_ok=True);path.rename(target)
    print('Previous outputs archived at',archive,flush=True)
    return str(archive)

def fingerprint():
    from feasibility.common import SETTINGS,REVISION
    from feasibility.catalogs import IMAGES,MEMBERS
    paths=[IMAGES,MEMBERS,*sorted((ROOT/'src/lenscluster').rglob('*.py')),
           *sorted(OUT.glob('*.py')),ROOT/'src/lenscluster/style.mplstyle']
    files={str(p.relative_to(ROOT)):digest(p) for p in paths}
    versions={}
    for name in ('jax','jaxlib','numpy','scipy','numpyro','arviz','astropy','jaxtronomy','lenstronomy','pandas','matplotlib','tqdm'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]='unavailable'
    identity=dict(revision=REVISION,settings=SETTINGS,files=files,versions=versions,
                  precision='float64',python=sys.version)
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest(),identity

class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,value):
        for stream in self.streams:stream.write(value);stream.flush()
    def flush(self):
        for stream in self.streams:stream.flush()
    def isatty(self):return False

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all',action='store_true')
    group.add_argument('--steps',nargs='+',choices=list(STEPS))
    parser.add_argument('--resume',action='store_true',help='Reuse compatible completed steps and retry failed ones')
    parser.add_argument('--restart',action='store_true',help='Archive existing generated outputs and start a fresh run')
    parser.add_argument('--cores',type=int,help='CPU budget, default min(8, available affinity CPUs)')
    parser.add_argument('--workers',type=int,help='Area processes and maximum concurrent starts, default min(4, cores)')
    parser.add_argument('--no-progress',action='store_true',help='Disable all progress bars')
    args=parser.parse_args(argv)
    if args.resume and args.restart:parser.error('--resume and --restart cannot be combined')
    from feasibility.runtime import configure,progress
    try:resources=configure(args.cores,args.workers,not args.no_progress)
    except ValueError as error:parser.error(str(error))
    from feasibility.common import write_json,read_json,jsonable
    OUT.mkdir(exist_ok=True);(OUT/'cosmography').mkdir(exist_ok=True);(OUT/'failures').mkdir(exist_ok=True)
    archived=archive_outputs() if args.restart else None
    key,identity=fingerprint();manifest=read_json(OUT/'manifest.json')
    if manifest:
        if not args.resume:parser.error('Existing manifest: use --resume, or choose a separate checkout/output directory')
        if manifest['fingerprint']!=key:parser.error('Checkpoint mismatch: inputs, code, configuration, Python, or dependencies changed. Preserve these outputs and start in a separate checkout.')
    else:
        manifest=dict(fingerprint=key,identity=identity,created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                      implementation_status='Written without execution; runtime checks are separate evidence',
                      host=platform.node(),platform=platform.platform(),cpu_count=os.cpu_count(),steps={})
    manifest.setdefault('resource_history',[]).append(dict(timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**resources))
    if archived:manifest['previous_run_archive']=archived
    write_json(OUT/'manifest.json',manifest)
    wanted=set(STEPS if args.all else args.steps)
    def dependencies(step):
        for dep in STEPS[step][2]:
            if dep not in wanted:wanted.add(dep);dependencies(dep)
    for step in list(wanted):dependencies(step)
    wanted.update(['report'])
    failed=False
    with (OUT/'run.log').open('a') as log:
        oldout,olderr=sys.stdout,sys.stderr
        sys.stdout,sys.stderr=Tee(oldout,log),Tee(olderr,log)
        try:
            import jax
            jax.config.update('jax_enable_x64',True)
            manifest['jax_devices']=[str(d) for d in jax.devices()]
            selected=[(step,definition) for step,definition in STEPS.items() if step in wanted]
            for step,(module,function,deps) in progress(selected,desc='Workflow',unit='step',position=0):
                previous=manifest['steps'].get(step,{})
                if args.resume and previous.get('status')=='complete' and step not in ('plots','report','residuals'):
                    hashes=previous.get('outputs',{})
                    # Mutable report/manifests are not scientific checkpoints.
                    if not all((OUT/p).exists() and digest(OUT/p)==sha for p,sha in hashes.items()):
                        raise RuntimeError(f'Checkpoint products changed for {step}; refusing unsafe reuse')
                    print(step,'reused compatible checkpoint',flush=True);continue
                blocked=[d for d in deps if manifest['steps'].get(d,{}).get('status')!='complete']
                if blocked:
                    manifest['steps'][step]=dict(status='skipped',reason='Unsuccessful dependencies: '+', '.join(blocked));failed=True
                    write_json(OUT/'manifest.json',manifest);continue
                print('START',step,flush=True);started=time.perf_counter()
                before={p:(p.stat().st_mtime_ns,p.stat().st_size) for p in OUT.rglob('*') if p.is_file() and 'archives' not in p.relative_to(OUT).parts}
                manifest['steps'][step]=dict(status='running',started=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
                write_json(OUT/'manifest.json',manifest)
                try:
                    fn=getattr(importlib.import_module('feasibility.'+module),function)
                    result=fn(free=True) if step.startswith('ratios-') else fn()
                    if isinstance(result,dict):
                        print(json.dumps(jsonable({k:v for k,v in result.items() if k not in ('traces','starts')}),indent=2),flush=True)
                    outputs={}
                    for p in OUT.rglob('*'):
                        if not p.is_file() or p.name in ('manifest.json','run.log') or '__pycache__' in p.parts or 'archives' in p.relative_to(OUT).parts:continue
                        if before.get(p)!=(p.stat().st_mtime_ns,p.stat().st_size):outputs[str(p.relative_to(OUT))]=digest(p)
                    manifest['steps'][step]=dict(status='complete',wall_seconds=time.perf_counter()-started,outputs=outputs,result=result,resources=resources)
                except Exception as error:
                    failed=True;trace=traceback.format_exc();print(trace,flush=True)
                    (OUT/'failures'/f'{step}.txt').write_text(trace)
                    manifest['steps'][step]=dict(status='failed',wall_seconds=time.perf_counter()-started,error=repr(error),resources=resources)
                write_json(OUT/'manifest.json',manifest)
                print('END',step,manifest['steps'][step]['status'],'wall seconds',time.perf_counter()-started,flush=True)
            # Refresh report once the final statuses have been saved.
            from feasibility.reporting import report
            report()
        finally:sys.stdout,sys.stderr=oldout,olderr
    return 1 if failed else 0

if __name__=='__main__':raise SystemExit(main())
