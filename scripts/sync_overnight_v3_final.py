"""Backfill the completed run's retained metrics after the SSH relay interruption."""
from pathlib import Path
import datetime,json
import wandb

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/rl_skills_v3_20261104/overnight'
RUN='jiguangbrt-university-of-dundee/rpent-pi05-skills/overnight20261104u257'

def main():
    previous=wandb.Api(timeout=20).run(RUN)
    before=int(previous.summary.get('train/update',previous.summary.get('update',256)))
    state=json.loads((REPORT/'relay-state.json').read_text())
    seen=set(state.get('evaluations',[]))
    if previous.summary.get('terminal_state')=='completed' and before>=807:
        print('Final metrics already synchronized; no duplicate backfill.')
        return
    rows=[json.loads(line) for line in (REPORT/'metrics.jsonl').read_text().splitlines()]
    assert len(rows)==550 and rows[0]['update']==258 and rows[-1]['update']==807
    assert len({r['update'] for r in rows})==550
    packets={}
    for row in rows:
        if row['update']<=before:continue
        metrics=dict(row['metrics'],update=row['update'])
        for actor in row['rollouts']:
            for key in ('seconds','inference_seconds','environment_seconds'):
                metrics[f"rollout/gpu{actor['physical_gpu']}_{key}"]=actor[key]
        packets[row['update']]=metrics
    eval_keys=[]
    for file in REPORT.glob('evaluation-*.json'):
        evaluation=json.loads(file.read_text())
        if 'metrics' not in evaluation:continue
        key=f"{evaluation['tag']}:{evaluation['update']}"
        if key in seen:continue
        packets.setdefault(evaluation['update'],dict(update=evaluation['update'])).update(evaluation['metrics'],evaluation_tag=evaluation['tag'])
        eval_keys.append(key)
    run=wandb.init(entity='jiguangbrt-university-of-dundee',project='rpent-pi05-skills',
                   id='overnight20261104u257',resume='must',mode='online',dir=str(REPORT),
                   settings=wandb.Settings(init_timeout=60,x_disable_stats=True))
    try:
        run.define_metric('update')
        for prefix in ('train/*','eval/*','rollout/*','coverage/*'):run.define_metric(prefix,step_metric='update')
        for update in sorted(packets):run.log(packets[update])
        result=json.loads((REPORT/'RESULTS.json').read_text())
        run.summary.update(dict(terminal_state='completed',final_update=807,additional_updates=550,
                           additional_native_steps=3349352,total_native_steps=4722269,
                           remote_status=result['status'],initialization_failures=result['initialization_failures'],
                           cloud_relay_interrupted_after_update=state['last_update'],
                           backfilled_training_updates=807-state['last_update']))
        for filename in ('RESULTS.md','learning-curves.png','evaluation-final.json'):
            run.save(str(REPORT/filename),base_path=str(REPORT),policy='now')
    finally:
        run.finish()
    final=wandb.Api(timeout=20).run(RUN)
    assert final.summary.get('update')==807 and final.summary.get('train/native_steps')==4722269
    proof=dict(passed=True,synchronized_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               cloud_url=final.url,previous_cloud_update=before,
               relay_interrupted_after_update=state['last_update'],backfilled_updates=807-state['last_update'],
               updates_submitted_this_pass=len([r for r in rows if r['update']>before]),
               added_evaluations=eval_keys,final_update=807,total_native_steps=4722269,
               terminal_state=final.summary.get('terminal_state'),server_archive_complete=True,
               note='Real-time relay stopped after 02:59:56+08:00. Retained server metrics were backfilled after delayed final-report trigger.')
    (REPORT/'final-cloud-sync.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(proof,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
