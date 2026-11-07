"""CPU budgeting and progress display, without importing numerical libraries."""
import os
import sys

def available_cpus():
    return sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else list(range(os.cpu_count() or 1))

def configure(cores=None,workers=None,progress_enabled=True):
    available=available_cpus()
    cores=min(8,len(available)) if cores is None else cores
    if cores<1 or cores>len(available):raise ValueError(f'--cores must be between 1 and {len(available)}')
    workers=min(4,cores) if workers is None else workers
    if workers<1 or workers>cores:raise ValueError('--workers must be between 1 and --cores')
    cpus=available[:cores]
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,cpus)
    # Before NumPy/JAX imports. Affinity bounds XLA's available CPU resources;
    # BLAS teams are smaller because optimization starts/chains can overlap.
    threads=max(1,cores//min(3,cores))
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]=str(threads)
    os.environ['JAX_NUM_CPU_DEVICES']=str(min(2,cores))
    os.environ['FEASIBILITY_WORKERS']=str(workers)
    os.environ['FEASIBILITY_CORES']=str(cores)
    os.environ['FEASIBILITY_PROGRESS']='1' if progress_enabled else '0'
    return dict(cpu_ids=cpus,cores=cores,workers=workers,blas_threads=threads,
                affinity_enforced=hasattr(os,'sched_setaffinity'),jax_cpu_devices=min(2,cores))

def worker_count():return int(os.environ.get('FEASIBILITY_WORKERS','1'))

def progress(iterable=None,**kwargs):
    from tqdm.auto import tqdm
    kwargs.setdefault('disable',os.environ.get('FEASIBILITY_PROGRESS','1')=='0')
    kwargs.setdefault('file',sys.__stderr__)
    kwargs.setdefault('dynamic_ncols',True)
    kwargs.setdefault('mininterval',.5)
    return tqdm(iterable,**kwargs)

def initialize_worker(cpu_queue):
    """Called by spawn before importing the JAX-heavy task modules."""
    cpus=cpu_queue.get()
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,cpus)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    os.environ['JAX_NUM_CPU_DEVICES']='1'
    os.environ['JAX_PLATFORMS']='cpu'
    os.environ['FEASIBILITY_PROGRESS']='0'
    os.environ['FEASIBILITY_WORKERS']='1'

def process_pool():
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    context=mp.get_context('spawn');count=worker_count();cpus=available_cpus()
    queue=context.Queue()
    for i in range(count):queue.put(cpus[i::count])
    return ProcessPoolExecutor(max_workers=count,mp_context=context,
                               initializer=initialize_worker,initargs=(queue,))

_area_model=None

def area_task(job):
    """One result returned to the parent, which alone owns checkpoint writes."""
    global _area_model
    import time
    from .model import Model
    from .area import one_area
    key,draw,z,report=job;begin=time.perf_counter()
    try:
        if _area_model is None:_area_model=Model()
        before=getattr(_area_model,'_feas_field_compile_seconds',0.)
        entry=one_area(_area_model,report,z)
        entry['field_compile_seconds']=getattr(_area_model,'_feas_field_compile_seconds',0.)-before
    except Exception as error:entry=dict(valid=False,reason=repr(error))
    entry.update(draw=draw,z=z,seconds=time.perf_counter()-begin)
    entry['execution_and_solver_setup_seconds']=entry['seconds']-entry.get('field_compile_seconds',0.)
    return key,entry
