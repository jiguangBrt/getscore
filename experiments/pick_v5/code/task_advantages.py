"""Task-local policy-gradient baselines for heterogeneous grasp difficulty."""
import numpy as np


def normalize_by_object(rows,std_floor=.25):
    names=sorted({r['object_type'] for r in rows});stats={}
    for name in names:
        group=[r for r in rows if r['object_type']==name]
        values=np.asarray([r['advantage'] for r in group],dtype=float)
        mean=float(values.mean());std=float(values.std());weight=len(rows)/(len(names)*len(group))
        for r in group:r['normalized_actor_advantage']=(r['advantage']-mean)/max(std,std_floor)*weight
        stats[name]=dict(chunks=len(group),raw_mean=mean,raw_std=std,scale=max(std,std_floor),task_weight=weight)
    assert all(np.isfinite(r['normalized_actor_advantage']) for r in rows)
    return stats
