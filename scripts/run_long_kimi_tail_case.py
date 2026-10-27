"""Run one not-yet-started tail case while the main paired workers continue."""
from argparse import ArgumentParser, Namespace
import json
from pathlib import Path
import run_long_benchmark as runner


def main():
    p=ArgumentParser(description=__doc__)
    p.add_argument('batch',type=Path)
    p.add_argument('--policy',choices=['pi05','gawm'],required=True)
    p.add_argument('--variant',choices=['original','swap'],required=True)
    p.add_argument('--task',type=int,choices=range(10),required=True)
    a=p.parse_args();batch=a.batch.resolve()
    planned=list(runner.cases(['kimi']))
    case=next(c for c in planned if c['variant']==a.variant and c['task']==a.task)
    name=runner.experiment_name(f'long-{a.variant}',a.task,'kimi',a.policy)
    output=batch/'cases'/name
    if (output/'benchmark_result.json').exists():return
    active=[]
    for f in (batch/'cases').glob('*/configuration.json'):
        c=json.loads(f.read_text())
        if c.get('run_status') in ['starting','running']:active.append(c)
    if len(active)>2:raise RuntimeError('Three simulation slots already occupied')
    target_index=(10 if a.variant=='swap' else 0)+a.task
    pending=[i for i,c in enumerate(planned) if not (batch/'cases'/runner.experiment_name(f"long-{c['variant']}",c['task'],'kimi',a.policy)/'benchmark_result.json').exists()]
    if pending and target_index-min(pending)<8:
        raise RuntimeError('Tail case too close to the main worker queue')
    for c in active:
        if c['planner']!='kimi' or c['policy']!=a.policy:continue
        current_index=(10 if c['variant']=='swap' else 0)+c['task']
        if target_index-current_index<8:raise RuntimeError('Tail case too close to active main worker')
    if (output/'configuration.json').exists():
        c=json.loads((output/'configuration.json').read_text())
        if c.get('run_status') in ['starting','running']:raise RuntimeError('Selected case already active')
    protocol=json.loads((batch/'protocol.json').read_text())
    args=Namespace(max_steps=protocol['max_episode_steps'],max_turns=protocol['max_turns'],timeout=protocol['planner_timeout_s'])
    runner.run_case(batch,case,a.policy,runner.environment(),args)


if __name__=='__main__':main()
