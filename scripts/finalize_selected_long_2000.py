"""Audit all selected results and build complete videos for continued episodes."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'RPent')]
from scripts import run_long_benchmark as runner

BATCH=ROOT/'outputs/selected-original-2000step-20261026'
TASKS={2:'打开炉灶并将摩卡壶放上去',3:'将黑碗放入柜子底层抽屉并关上',4:'白杯放左盘，黄白杯放右盘',
       5:'将书放入收纳架后侧隔间',6:'白杯放盘中，巧克力布丁放盘右侧',8:'将两个摩卡壶放在炉灶上',9:'将黄白杯放入微波炉并关门'}


def hash_artifacts(directory):
    episode=next(directory.glob('trajectory/raw/libero/*/task_*/seed_*/episode_*'))
    paths={p.name:p for p in [directory/'episode.mp4',directory/'states.json',
        *directory.glob('transcript_*.json'),episode/'transitions.npz',episode/'proposals.npz']}
    verification=json.loads((directory/'verification.json').read_text())
    for name,digest in verification['sha256'].items():
        if runner.sha(paths[name])!=digest:raise RuntimeError(f'Artifact changed: {directory.name}/{name}')
    return verification,episode


def video_info(path,expected):
    import imageio.v2 as imageio
    with imageio.get_reader(path) as reader:
        meta=reader.get_meta_data();count=0
        for frame in reader:
            assert frame.shape==(256,256,3),(path,frame.shape)
            count+=1
    assert count==expected,(path,count,expected)
    assert abs(meta['fps']-20)<1e-6
    return dict(path=str(path),frames=count,fps=meta['fps'],all_frames_decoded=True)


def audit_one(item):
    import numpy as np
    from robots.libero.flywheel import LIBERO_SPEC
    from rpent.flywheel.episode import validate_episode
    source=Path(item['source_run']);directory=BATCH/'cases'/source.name
    original=json.loads((source/'benchmark_result.json').read_text())
    row=json.loads((directory/'benchmark_result.json').read_text())
    assert not original['success'] and row['error'] is None
    assert row['task']==item['task'] and row['planner']==item['planner'] and row['policy']==item['policy']
    _,original_episode=hash_artifacts(source)
    verification,episode=hash_artifacts(directory)
    metadata=validate_episode(episode,spec=LIBERO_SPEC)
    with np.load(episode/'transitions.npz') as data:
        count=len(data['actions']);native=bool(data['terminated'].any())
        truncated=bool(data['truncated'].any())
        final_image=data['main_images'][0]
        final_wrist=data['wrist_images'][0]
    states=json.loads((directory/'states.json').read_text())['steps']
    assert native==row['success']==bool(states[-1]['terminated'])==metadata['is_success']
    assert count==row['actions']==row['new_actions']==verification['frames']
    assert metadata['step_count']==count
    assert row['total_episode_actions']==row['resumed_from_step']+count<=2000
    recovery=json.loads((directory/'recovery.json').read_text())
    if row['execution_mode']=='resume':
        assert recovery['initial_images_exact'] and recovery['final_images_exact']
        assert recovery['first_replay_mismatch_step'] is None
        assert recovery['max_eef_component_error_m']<=1e-6
        assert row['resumed_from_step']==original['actions']==1000
        with np.load(original_episode/'transitions.npz') as data:
            assert np.array_equal(final_image,data['main_images'][-1])
            assert np.array_equal(final_wrist,data['wrist_images'][-1])
        assert (directory/'kimi_context_recovery.json').is_file()
    else:assert row['resumed_from_step']==0
    video=video_info(directory/'episode.mp4',count)
    transcript=json.loads(next(directory.glob('transcript_*.json')).read_text())
    stop='环境原生成功' if native else '累计2000步用尽' if truncated and row['total_episode_actions']==2000 else '规划器主动结束'
    result=dict(experiment=directory.name,task=item['task'],task_zh=TASKS[item['task']],planner=item['planner'],policy=item['policy'],
        old_success=original['success'],old_actions=original['actions'],new_success=native,execution_mode=row['execution_mode'],
        added_actions=count,total_episode_actions=row['total_episode_actions'],stop_reason=stop,
        source_run=str(source),directory=str(directory),recovery_reason=recovery['reason'],
        source_artifact_hashes_unchanged=True,native_success_agrees=True,raw_trajectory_valid=True,video=video,
        planner_completion_error=row.get('planner_completion_error'),finish=transcript.get('finish'))
    print('AUDIT PASS',directory.name,'success='+str(native),'total='+str(row['total_episode_actions']),flush=True)
    return result


def complete_video(row):
    import imageio_ffmpeg
    directory=Path(row['directory']);source=Path(row['source_run'])
    full=directory/'episode-full.mp4'
    if not full.exists():
        command=[imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error',
            '-i',str(source/'episode.mp4'),'-i',str(directory/'episode.mp4'),
            '-filter_complex','[0:v:0][1:v:0]concat=n=2:v=1:a=0[out]',
            '-map','[out]','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(full)]
        subprocess.run(command,check=True,timeout=180)
    info=video_info(full,row['total_episode_actions'])
    info.update(source_prefix=str(source/'episode.mp4'),continuation_suffix=str(directory/'episode.mp4'),
        sha256=runner.sha(full),role='derived full replay; source videos preserved')
    runner.save(directory/'full-video-verification.json',info)
    row['full_video']=str(full)
    return info


def main():
    selection=json.loads((BATCH/'selection.json').read_text())
    assert selection['status']=='complete' and len(selection['cases'])==13
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows=list(pool.map(audit_one,selection['cases']))
    full=[]
    for row in rows:
        if row['execution_mode']=='resume':full.append(complete_video(row))
    successes=sum(r['new_success'] for r in rows)
    groups={}
    for row in rows:
        group=groups.setdefault(row['planner']+'-'+row['policy'],dict(total=0,successes=0,failures=0))
        group['total']+=1;group['successes']+=int(row['new_success']);group['failures']+=int(not row['new_success'])
    audit=dict(verified_at=runner.now(),complete=True,total=len(rows),successes=successes,failures=len(rows)-successes,
        resumed=sum(r['execution_mode']=='resume' for r in rows),restarted=sum(r['execution_mode']=='restart' for r in rows),
        all_native_success_agrees=True,all_action_budgets_within_2000=True,all_video_frames_decoded=True,
        selected_original_artifact_hashes_unchanged=True,groups=groups,full_replay_videos=full,results=rows)
    assert (audit['total'],successes,audit['resumed'])==(13,6,3)
    runner.save(BATCH/'final-verification.json',audit)
    fields=['experiment','task','task_zh','planner','policy','old_success','old_actions','execution_mode','new_success',
            'added_actions','total_episode_actions','stop_reason','replay_video','continuation_video','recovery_reason','planner_completion_error']
    with (BATCH/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader()
        for row in rows:
            writer.writerow({**{k:row.get(k) for k in fields},'replay_video':row.get('full_video',row['video']['path']),
                'continuation_video':row['video']['path'] if row['execution_mode']=='resume' else ''})
    lines=['# 13 项 2000 步实验结果',f'核验时间：{audit["verified_at"]}',
        f'全部完成：13/13；成功 {successes} 项，未成功 {13-successes} 项。3 项从历史终点续跑，10 项从头重跑。',
        '这 13 项的原始结果均未成功。新旧结果独立保存；本次结果不覆盖原始 120 次评测。',
        'Kimi 全部 11 项执行后才执行 2 项 GPT。Kimi 为 kimi-code/k3.1-pro / high；GPT 为 gpt-6-astra / low。',
        '', '| 任务 | 规划器＋动作模型 | 原结果 | 执行方式 | 新结果 | 本次执行步数 | 累计步数 | 停止原因 | 视频 |',
        '|---|---|---|---|---|---:|---:|---|---|']
    for row in rows:
        planner='Kimi' if row['planner']=='kimi' else 'GPT-6'
        policy='π0.5' if row['policy']=='pi05' else 'GAWM'
        mode='从1000步续跑' if row['execution_mode']=='resume' else '从头重跑'
        links=f'[完整回放]({row["full_video"]}) · [追加段]({row["video"]["path"]})' if row.get('full_video') else f'[回放]({row["video"]["path"]})'
        lines.append(f'| {row["task"]}：{row["task_zh"]} | {planner}＋{policy} | 未成功 | {mode} | {"成功" if row["new_success"] else "未成功"} | {row["added_actions"]} | {row["total_episode_actions"]} | {row["stop_reason"]} | {links} |')
    lines.extend(['','## 如何理解结果','',
        '- Kimi＋π0.5 的任务4保留原来的1000步环境和会话，再执行265步后获得原生成功，累计1265步。这是本批次直接验证延长原episode预算有效的案例。',
        '- 其余5项成功来自从头重跑。初始画面和原记录不一致，不能把这些成功全部归因于从1000步增加到2000步；其中两项在新episode的439和444步就已成功。',
        '- 3项续跑中另两项（任务6的π0.5和GAWM组合）追加1000步后仍未成功。',
        '- GPT＋π0.5 的任务5和8分别在957、1739步由规划器主动结束，并非用尽2000步。原始finish说明保存在各自transcript中。',
        '- Kimi＋π0.5 任务3的原生episode已完整执行2000步后，旧版规划器等待触发2400秒墙钟超时。该原始错误保留为planner_completion_error，原始transcript不改动；依据原生截断和通过哈希、视频核验的完整轨迹保留未成功结果，避免重复执行。',
        '- Kimi＋GAWM 任务2的一次1463步墙钟中断保存在incomplete目录，不计作任务失败或有效评测；随后完整重跑2000步仍未成功。',
        '', '## 核验','',
        '13项均通过：原生成功标志与结果一致、原始轨迹校验通过、全部视频逐帧解码、本次视频帧数等于本次动作数、episode累计不超过2000步。所选13项原实验的既有产物哈希均与原verification.json一致。3项续跑的完整拼接视频也已逐帧核对累计帧数。',
        '',f'[机器可读核验记录]({BATCH/"final-verification.json"}) · [CSV对比表]({BATCH/"comparison.csv"})'])
    (BATCH/'report.md').write_text('\n\n'.join(lines[:6])+'\n\n'+'\n'.join(lines[6:])+'\n')
    selection.pop('model_calls_for_new_experiments',None)
    selection.update(status='complete',updated_at=runner.now(),final_verified_at=audit['verified_at'],completed=13,successes=successes,failures=13-successes)
    for item,row in zip(selection['cases'],rows):
        item.update(status='complete',new_success=row['new_success'],execution_mode=row['execution_mode'],
            added_actions=row['added_actions'],total_episode_actions=row['total_episode_actions'],new_result_directory=row['directory'])
    runner.save(BATCH/'selection.json',selection)
    print('FINAL VERIFIED',len(rows),'successes',successes,'failures',13-successes,flush=True)


if __name__=='__main__':main()
