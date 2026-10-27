"""Write the final paired task table and artifact audit without rerunning episodes."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_viewer.naming import case_directories


def main():
    p=argparse.ArgumentParser();p.add_argument('batch',type=Path);args=p.parse_args();batch=args.batch.resolve()
    results=[json.loads((directory/'benchmark_result.json').read_text())
             for directory in case_directories(batch) if (directory/'benchmark_result.json').is_file()]
    expected_gawm_sha=json.loads((batch/'gawm-policy-metadata.json').read_text())['checkpoint_sha256']
    labels_path=Path(__file__).resolve().parents[1]/'data/libero-original-pro-task-swap-comparison-20261023/tasks_zh.json'
    labels={(e['suite'],e['task_id']):e['language_zh'] for e in json.loads(labels_path.read_text())}
    lookup={(r['variant'],r['task'],r['planner'],r['policy']):r for r in results}
    combinations=[('none','pi05'),('none','gawm'),('gpt6','pi05'),('gpt6','gawm')]
    trigger_path=batch/'kimi_trigger.json'
    kimi_triggered=json.loads(trigger_path.read_text()).get('triggered',False) if trigger_path.exists() else False
    kimi_present=any(r['planner']=='kimi' for r in results) or kimi_triggered
    if kimi_present:combinations.extend([('kimi','pi05'),('kimi','gawm')])
    fields=['variant','task','task_zh']
    for planner,policy in combinations:
        fields.extend([f'{planner}_{policy}_success',f'{planner}_{policy}_actions',f'{planner}_{policy}_error',f'{planner}_{policy}_video'])
    rows=[];initial_audit=[]
    for variant,suite in [('original','libero_10'),('swap','libero_10_swap')]:
        for task in range(10):
            row=dict(variant=variant,task=task,task_zh=labels[(suite,task)])
            hashes=set();states=[]
            for planner,policy in combinations:
                key=f'{planner}_{policy}';r=lookup.get((variant,task,planner,policy))
                if r is None:continue
                row.update({key+'_success':int(r['success']),key+'_actions':r['actions'],key+'_error':r['error'] or '',key+'_video':os.path.relpath(r['video'],batch)})
                config=json.loads((Path(r['directory'])/'configuration.json').read_text());hashes.add(config['init_files']['sha256'])
                assert config['policy_status']['num_action_chunks']==(8 if policy=='gawm' else 5)
                if policy=='gawm':assert config['policy_status']['metadata']['checkpoint_sha256']==expected_gawm_sha
                recorded=json.loads((Path(r['directory'])/'states.json').read_text())['steps']
                assert all((s.get('command') or {}).get('action')!='reset' for s in recorded),r['name']
                trace=recorded[0]['state']
                if planner=='gpt6':
                    assert config['model']=='gpt-6-astra' and config['reasoning_effort']=='low'
                    assert list(Path(r['directory']).glob('codex_*.txt.stream.jsonl'))
                elif planner=='kimi':
                    settings=json.loads((Path(r['directory'])/'kimi_session.json').read_text())['confirmed_settings']
                    assert settings['model']['currentValue']=='kimi-code/k3.1-pro'
                    assert settings['thinking']['currentValue']=='high'
                states.append({k:trace[k] for k in ['robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos']})
                assert r['actions']<=1000,r['name']
                assert r['frames']==r['actions'],r['name']
                assert Path(r['video']).is_file(),r['name']
            if hashes:assert len(hashes)==1,(variant,task)
            if states:
                import numpy as np
                a=np.concatenate([np.asarray(v).reshape(-1) for v in states[0].values()])
                max_diff=max(float(np.max(np.abs(a-np.concatenate([np.asarray(v).reshape(-1) for v in st.values()])))) for st in states)
                assert max_diff<1e-8,(variant,task,max_diff)
                initial_audit.append(dict(variant=variant,task=task,cases=len(states),same_init_file=True,same_state_index=True,max_robot_state_difference=max_diff))
            rows.append(row)
    with (batch/'task-comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    group=defaultdict(lambda:dict(successes=0,total=0,errors=0,policy_queries=0,actions=0,elapsed_s=0.0))
    for r in results:
        g=group[(r['variant'],r['planner'],r['policy'])]
        g['total']+=1;g['successes']+=int(r['success']);g['errors']+=int(bool(r['error']));g['policy_queries']+=r['policy_queries']
        g['actions']+=r['actions'];g['elapsed_s']+=r['elapsed_s']
    with (batch/'group-comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,['variant','planner','policy','successes','total','success_rate','planner_errors','mean_actions','mean_elapsed_s','policy_queries'])
        w.writeheader()
        for (variant,planner,policy),g in sorted(group.items()):
            w.writerow(dict(variant=variant,planner=planner,policy=policy,successes=g['successes'],total=g['total'],
                success_rate=g['successes']/g['total'],planner_errors=g['errors'],mean_actions=round(g['actions']/g['total'],2),
                mean_elapsed_s=round(g['elapsed_s']/g['total'],2),policy_queries=g['policy_queries']))
    audit=dict(checked_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),cases=len(results),complete=len(results)==(120 if kimi_present else 80),
        expected=120 if kimi_present else 80,all_cases_have_video=True,all_videos_match_action_counts=True,
        all_action_counts_within_1000=True,no_episode_resets=True,requested_model_settings_verified=True,paired_initialization=initial_audit,
        initialization_note='All groups use the same official init file and state index; robot state agrees within 1e-8. RGB is not asserted byte-identical across simulator processes.')
    (batch/'final-verification.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    lines=['Long 原版 / Swap 模型对比评测报告',f'统计时间：{audit["checked_at"]}',f'已归档试验：{len(results)}/{audit["expected"]}',
        '每任务每组单次尝试，seed 0、官方初始状态 0；最多 1000 个原生动作；禁止重置。',
        '原版使用与标准 LIBERO 逐字节一致的任务定义和初始化文件；两种场景统一在本机 PRO / robosuite 1.5.2 / MuJoCo 3.3.0 环境执行。',
        'GPT：gpt-6-astra / low reasoning；Kimi（追加时）：kimi-code/k3.1-pro / 1M / high thinking。',
        '各规划器使用相同 RPent 工具和空的外部任务记忆；保留 RPent 原有通用操作提示。',
        '纯策略直接执行完整任务提示词，π0.5 每次执行 5 个动作，GAWM 每次执行 8 个动作。',
        '沿用之前 GPT+GAWM 接入方式：GAWM 的策略调用固定使用本任务的完整标准提示词；π0.5 可接受子指令。规划器仍可使用通用运动/夹爪工具纠错。',
        '成功只以环境原生 terminated 为准，模型自行声称成功不计分。','',
        '场景       规划器  策略   成功/全部  成功率  规划器错误']
    for (variant,planner,policy),g in sorted(group.items()):
        lines.append(f'{variant:10s} {planner:6s} {policy:5s} {g["successes"]:2d}/{g["total"]:<2d}   {g["successes"]/g["total"]:6.1%}   {g["errors"]}')
    lines.extend(['','逐任务结果：task-comparison.csv；分组统计：group-comparison.csv；详细记录：results.csv、summary.json。',
        '各任务目录直接位于 outputs/long-{original|swap}-{0..9}-{none|gpt6|kimi}-{pi05|gawm}/，其中包含配置、日志、任务审计 JSON、相机/深度观测、episode.mp4 与原始 trajectory/。cases/ 保留兼容链接。',
        'GPT 的公开输出与工具交互见 codex_*.txt 及 *.stream.jsonl；Kimi 对应 kimi.txt、kimi_acp.stream.jsonl。',
        '核验记录：各案例 verification.json 和总目录 final-verification.json。',
        '此结果是各任务单次运行的配对比较，未进行多随机种子重复统计。'])
    if kimi_present:
        lines.append('Kimi 与 GPT 使用同一 RPent 任务提示和工具，但分别通过 Kimi Code ACP 与 Codex SDK 接入；模型和 CLI 自带行为同时变化。Kimi ACP 不暴露模型回合数限制，两者均受 1200 秒规划时间和 1000 动作上限约束。')
    if (batch/'kimi-quota-recovery.json').exists():
        lines.append('Kimi 额度中断：10 月 25 日约 17:20 达到 5 小时调用上限，ACP 2.3.1 静默返回 end_turn。一项 264 动作的中断试验和 15 项零动作会话已归档到 incomplete/kimi-quota-interruption-20261025-1720，不计为模型失败；额度恢复后仅续跑这些未完成项。')
    if (batch/'resume-preflight-20261025.json').exists():
        lines.append('续跑说明：10 月 23 日 Codex 额度耗尽。21 项未执行动作，另有原版 GPT+π0.5 task 8 在 65 个脚本动作后被中断；这些基础设施中断记录保留在 incomplete/，不计为模型失败。10 月 25 日额度恢复后补跑，正常完成的任务失败未重试。')
    if (batch/'output-layout-audit.json').exists():
        lines.append('目录兼容说明：对照两份历史试验核验观测目录和顶层文件；补充缺失的任务审计时明确标注由已有记录生成。原始 transcript/states/CLI 事件保持不变；纯策略与 Kimi 另提供带真实来源标记的网站兼容事件。失败试验不导出成功 recipe。')
    if (batch/'model-migration.json').exists():
        migration=json.loads((batch/'model-migration.json').read_text())
        lines.append('Kimi 模型更换：按用户要求停止旧 K3，主结果仅统计 K3.1 Pro / high。此前 K3 的 6 项完整结果与 3 项中断记录单独保留在 '+migration['old_model_archive']+'，不混入新模型成功率。')
    (batch/'report.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:25]))


if __name__=='__main__':main()
