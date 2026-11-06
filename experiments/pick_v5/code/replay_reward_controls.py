"""Re-score recorded physical controls under a reward-only ablation."""
import argparse
import hashlib
import json
from pathlib import Path
from pick_reward import RewardLedger


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--physical',type=Path,required=True);p.add_argument('--demos',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();cfg=json.loads(a.config.read_text())
    paths=[(p,'control') for p in a.physical.glob('*/native-trace.jsonl')]
    chosen={}
    for row in json.loads(a.demos.read_text())['results']:
        if row['success'] and row['type'] not in chosen:chosen[row['type']]=Path(row['data']).parent/'native-trace.jsonl'
    paths.extend((p,'positive') for p in chosen.values());rows=[]
    for path,role in paths:
        reset=json.loads((path.parent/'reset.json').read_text());states=[json.loads(x) for x in path.read_text().splitlines()]
        ledger=RewardLedger(cfg,'pick',reset['metrics'])
        for state in states:_,_,_,last=ledger.advance(state)
        assert all(last[k]==states[-1][k] for k in ('skill_success','failure','timeout','termination_reason'))
        if role=='positive':assert last['skill_success'] and last['episode_reward']>0
        rows.append(dict(path=str(path),role=role,type=reset['spec']['type'],success=last['skill_success'],
                         old_return=states[-1]['episode_reward'],new_return=last['episode_reward'],
                         trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert len(chosen)==len(cfg['object_types'])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(dict(passed=True,config=str(a.config),rows=rows,
        note='reward recomputation on unchanged physically recorded states; not new policy episodes'),indent=2)+'\n')
