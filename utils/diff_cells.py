import difflib
from IPython import get_ipython
def _clean(src: str):
    out=[]
    for l in src.splitlines():
        if l.strip().startswith("#"): continue
        if not l.strip(): continue
        out.append(l.rstrip())
    return out
def count_cell_diff(cpu_cell_idx: int, gpu_cell_idx: int):
    In = get_ipython().user_ns.get('In', [])
    a=_clean(In[cpu_cell_idx]); b=_clean(In[gpu_cell_idx])
    diff=list(difflib.unified_diff(a,b,fromfile=f"Cell{cpu_cell_idx}",tofile=f"Cell{gpu_cell_idx}",lineterm=""))
    changes=[l for l in diff if (l.startswith("+") or l.startswith("-")) and not l.startswith(("+++","---","@@"))]
    return len(changes), diff
