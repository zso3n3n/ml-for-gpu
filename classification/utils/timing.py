import os, gc, time
def set_cpu_threads(n=8): os.environ.setdefault("OMP_NUM_THREADS", str(n))
def _sync(use_gpu: bool):
    if use_gpu:
        import cupy as cp
        cp.cuda.Device().synchronize()
def run_timed(label, fn, use_gpu, *a, **k):
    gc.collect(); _sync(use_gpu)
    t0=time.perf_counter(); out=fn(*a, **k); _sync(use_gpu)
    dt=time.perf_counter()-t0; print(f"{label}: {dt:.3f}s"); return out, dt
