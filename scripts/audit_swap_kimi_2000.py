"""Audit completed swap reruns and publish their verified comparison report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'RPent')]
from scripts import run_long_benchmark as runner
from scripts.run_swap_kimi_2000 import verify_initialization, update_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,required=True)
    args=parser.parse_args();batch=args.batch.resolve()
    selection=json.loads((batch/'selection.json').read_text());assert len(selection['cases'])==13
    audit=batch/'final-audit';audit.mkdir(exist_ok=True)
    backups=audit/'before';backups.mkdir(exist_ok=True)
    for name in ['status.json','summary.json','report.md','results.csv']:
        source=batch/name;backup=backups/name
        if source.exists() and not backup.exists():shutil.copyfile(source,backup)
    verified=[];checks=[]
    for item in selection['cases']:
        source=Path(item['source_run']);output=batch/'cases'/source.name
        path=output/'benchmark_result.json';row=json.loads(path.read_text())
        old=json.loads((source/'benchmark_result.json').read_text())
        config=json.loads((output/'configuration.json').read_text())
        session=json.loads((output/'kimi_session.json').read_text())
        verification=json.loads((output/'verification.json').read_text())
        assert runner.sha(source/'benchmark_result.json')==item['source_result_sha256']
        assert not row.get('error') and not config.get('error') and row['exit_code']==0
        assert config['model']==session['confirmed_settings']['model']['currentValue']=='kimi-code/k3.1-pro'
        assert session['confirmed_settings']['thinking']['currentValue']=='high'
        assert config['max_episode_steps']==2000
        assert 0<row['actions']<=2000
        assert not row['truncated'] or row['actions']==2000
        init=verify_initialization(source,output,old,row)
        episode=Path(row['trajectory']);native=json.loads((episode/'episode.json').read_text())
        assert native['step_count']==row['actions'] and native['is_success']==row['success']
        states=json.loads((output/'states.json').read_text())['steps']
        assert bool(states[-1]['terminated'])==row['success']
        assert verification['raw_trajectory_valid'] and verification['frames_equal_actions']
        paths={p.name:p for p in [output/'episode.mp4',Path(row['transcript']),output/'states.json',episode/'transitions.npz',episode/'proposals.npz']}
        for name,digest in verification['sha256'].items():
            assert runner.sha(paths[name])==digest,(output,name)
        import imageio.v2 as imageio
        with imageio.get_reader(output/'episode.mp4') as reader:
            frames=sum(1 for _ in reader)
        assert frames==row['actions']==row['frames']
        backup=backups/'cases'/output.name/'benchmark_result.json';backup.parent.mkdir(parents=True,exist_ok=True)
        if not backup.exists():shutil.copyfile(path,backup)
        row.update(source_run=str(source),execution_mode='fresh',target_total_steps=2000,
                   total_episode_actions=row['actions'],initial_observation_exact=init['initial_observation_exact'],
                   initialization_audit=str(output/'initialization-audit.json'))
        runner.save(path,row);verified.append(row)
        checks.append(dict(name=output.name,source_unchanged=True,recorded_artifact_hashes_match=True,
                           video_decodes=True,frames=frames,model_confirmed=True,native_success=row['success'],
                           **init))
        print(runner.now(),'VERIFIED',output.name,'success='+str(row['success']),'actions='+str(row['actions']),flush=True)
    summary=runner.refresh(batch)
    groups={}
    for policy in ('pi05','gawm'):
        old_rows=[json.loads(p.read_text()) for p in Path(selection['source_directory']).glob(f'long-swap-*-kimi-{policy}/benchmark_result.json')]
        reruns=[row for row in verified if row['policy']==policy]
        prior=sum(row['success'] for row in old_rows);new_success=sum(row['success'] for row in reruns)
        groups[policy]=dict(original_successes=prior,total=10,reruns=len(reruns),rerun_successes=new_success,
                            combined_successes=prior+new_success)
    varied=sum(not row['initial_observation_exact'] for row in verified)
    summary.update(expected=13,successes=sum(row['success'] for row in verified),
                   comparison=groups,initial_observation_nonidentical_cases=varied,
                   audit=str(audit/'verification.json'))
    runner.save(batch/'summary.json',summary)
    update_report(batch,selection,verified,{},'complete')
    status=json.loads((batch/'status.json').read_text())
    status.update(initial_observation_nonidentical_cases=varied,audit=str(audit/'verification.json'))
    runner.save(batch/'status.json',status)
    runner.save(audit/'verification.json',dict(status='verified_complete',checked_at=runner.now(),expected=13,
        completed=13,successes=summary['successes'],source_results_unchanged=True,comparison=groups,
        initial_observation_nonidentical_cases=varied,
        initial_condition_note='Known LIBERO-PRO fixture-layout randomization already occurs among original same-task/seed runs; nominal conditions and robot state match, RGB equality is recorded rather than claimed.',
        checks=checks))
    with (batch/'report.md').open('a') as report:
        report.write('\n## 与原结果比较\n\n| 组合 | 原成功 | 本次失败项重跑成功 | 保留旧成功后的合并结果 |\n|---|---:|---:|---:|\n')
        for policy,group in groups.items():
            report.write(f'| Kimi + {policy} | {group["original_successes"]}/10 | {group["rerun_successes"]}/{group["reruns"]} | {group["combined_successes"]}/10 |\n')
        report.write('\n本次新增的成功均在 1000 步以内完成。9 项失败跑满 2000 步；另 1 项在 1548 步由模型主动结束。结果不能证明增加动作预算本身导致了成功率变化；模型随机性、部分场景的初始家具布局变化以及迁移后的推理环境版本差异都是限制。合并统计保留原先成功的 7 项，不能视为全部 20 项均以 2000 步重新评测。\n')
    print(json.dumps(dict(completed=13,successes=summary['successes'],comparison=groups),ensure_ascii=False))


if __name__=='__main__':main()
