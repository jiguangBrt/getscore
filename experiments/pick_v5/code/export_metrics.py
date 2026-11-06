"""Small reviewable summary; full native traces remain in their run directories."""
import argparse
import json
from pathlib import Path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();result={}
    for name in ('pick-v5-smoke-20261106-a','pick-v5-smoke-20261106-d',
                 'pick-v5-warmup-20261106-e','pick-v5-warmup-targeted-20261106-g',
                 'pick-v5-pilot100-20261106-f','pick-v5-pilot800-20261106-h',
                 'pick-v5-pilot701-20261106-k','pick-v5-hold-ablation100-20261106-l',
                 'pick-v5-balanced200-20261106-m'):
        run=a.runs_root/name
        if not run.exists():continue
        record=dict(path=str(run),evaluations=[])
        for kind in ('status','effective-config','initial-source','resume-source'):
            path=run/f'{kind}.json'
            if path.exists():record[kind]=json.loads(path.read_text())
        for kind in ('metrics','episodes'):
            path=run/f'{kind}.jsonl'
            if path.exists():
                rows=[json.loads(x) for x in path.read_text().splitlines()]
                if kind=='episodes':
                    rows=[{k:r[k] for k in ('episode_id','update','seed','object_type','native_steps','episode_reward','success','failure','timeout')} for r in rows]
                record[kind]=rows
        for path in sorted(run.glob('evaluation-*.json')):
            r=json.loads(path.read_text())
            if isinstance(r,dict) and 'metrics' in r:
                record['evaluations'].append(dict(tag=path.stem.removeprefix('evaluation-'),
                    update=r['update'],metrics=r['metrics'],reused_from=r.get('reused_from')))
        result[name]=record
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
