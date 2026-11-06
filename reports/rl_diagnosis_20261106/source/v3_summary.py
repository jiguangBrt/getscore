"""Always-write result artifact, including interrupted/failed runs."""
import json,time,sys
from pathlib import Path

def summary(out):
    out=Path(out)
    def read(p,default):
        try:return json.loads(p.read_text())
        except (OSError,ValueError):return default
    status=read(out/'status.json',{});supervisor=read(out/'supervisor-status.json',{});meta=read(out/'run-meta.json',{})
    display_state=supervisor.get('state') if supervisor.get('state') in ['failed','stopped','hard_deadline','stalled'] else status.get('state',supervisor.get('state','starting'))
    rows=[]
    if (out/'metrics.jsonl').exists():
        for l in (out/'metrics.jsonl').read_text().splitlines():
            try:rows.append(json.loads(l))
            except ValueError:pass
    rows=list({r['update']:r for r in rows}.values())
    rows.sort(key=lambda r:r['update'])
    evaluations=[]
    for p in sorted(out.glob('evaluation-*.json')):
        if p.name=='evaluation-cases.json':continue
        e=read(p,{})
        if 'metrics' in e:evaluations.append(dict(file=p.name,tag=e['tag'],update=e['update'],metrics=e['metrics']))
    checkpoints=[dict(file=p.name,bytes=p.stat().st_size) for p in sorted((out/'checkpoints').glob('*.pt'))]
    episodes=[]
    if (out/'episodes.jsonl').exists():
        for line in (out/'episodes.jsonl').read_text().splitlines():
            try:episodes.append(json.loads(line))
            except ValueError:pass
    committed={(r['update'],r.get('trainer_pid')) for r in rows}
    episodes=[e for e in episodes if (e.get('update'),e.get('trainer_pid')) in committed]
    outcomes={skill:{key:sum(bool(e.get(key)) for e in episodes if e['skill']==skill) for key in ('success','failure','timeout')} for skill in ('pick','place','upright')}
    discards=[];budget_truncations=[]
    if (out/'events.jsonl').exists():
        for line in (out/'events.jsonl').read_text().splitlines():
            try:
                row=json.loads(line)
                if row['kind']=='restart_discards_unfinished':discards.append(row)
                if row['kind']=='run_budget_truncated':budget_truncations.append(row)
            except ValueError:pass
    migration=read(out/'MIGRATION.json',{});continuation=read(out/'CONTINUATION.json',{});wb=read(out/'wandb.json',{});cloud=read(out/'wandb-cloud.json',{})
    result=dict(continuation=continuation,wandb=wb,wandb_cloud=cloud,additional_updates=(rows[-1]['update']-continuation.get('update',0)) if rows else 0,additional_native_steps=(rows[-1]['native_steps']-continuation.get('native_steps',0)) if rows else 0,resource_migration=migration,initialization_failures=sum(r['metrics'].get('train/reset_errors',0) for r in rows),budget_truncated=budget_truncations[-1]['count'] if budget_truncations else None,outcomes=outcomes,restart_truncated=sum(len(r['active']) for r in discards),updated_at=time.time(),state=display_state,status=status,supervisor=supervisor,run_meta=meta,completed_updates=rows[-1]['update'] if rows else 0,native_steps=rows[-1]['native_steps'] if rows else 0,evaluations=evaluations,checkpoints=checkpoints,videos=[str(p.relative_to(out)) for p in sorted((out/'videos').glob('*.mp4'))],wandb_offline_dirs=[str(p) for p in (out/'wandb').glob('offline-run-*')])
    tmp=out/'RESULTS.json.tmp';tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');tmp.replace(out/'RESULTS.json')
    lines=['# π0.5 三技能训练结果','',f"更新时间：{time.strftime('%Y-%m-%d %H:%M:%S %Z')}",f"状态：{display_state}",f"完成更新：{result['completed_updates']}；原生动作步数：{result['native_steps']}",'','受控续训：物理 GPU 0、1、2、3，共享同一 action-expert LoRA + critic；24环境；夜间续训硬截止为北京时间2026-11-05 09:00，08:50起预留最终评测。',f"W&B 模式：{wb.get('mode','starting')}；云端：{cloud.get('url') or wb.get('url') or '本机中继待连接'}；本地日志保留在 wandb/。",'','## 评测','', '| 时点 | 更新 | 抓取 | 放置 | 扶正 | 有效/尝试 |','|---|---:|---:|---:|---:|---:|']
    for e in sorted(evaluations,key=lambda e:e['update']):
        m=e['metrics'];valid=sum(m.get(f'eval/{s}_valid',0) for s in ['pick','place','upright']);invalid=sum(m.get(f'eval/{s}_invalid_reset',0) for s in ['pick','place','upright'])
        lines.append(f"| {e['tag']} | {e['update']} | {m['eval/pick_success']:.1%} | {m['eval/place_success']:.1%} | {m['eval/upright_success']:.1%} | {valid}/{valid+invalid} |")
    lines+=['','成功率分母为有效物理初态；初始化失败另列，不计作策略成功。评测是自定义技能环境，不能当作 LIBERO 原基准或 RPent 全流程成绩。','','## Checkpoints','']
    for p in checkpoints:lines.append(f"- [{p['file']}](checkpoints/{p['file']}) ({p['bytes']/2**20:.1f} MiB)")
    lines+=['','Checkpoint 包含 LoRA、value head、优化器、RNG、配置和基础模型哈希。推理时需与指定原始基础权重组合；不可直接当作完整基础模型。活动环境跨采样批次继续；进程重启时丢弃未完成环境轨迹并单独记录，不能逐位续跑。','','## 视频','']
    for p in result['videos'][-18:]:lines.append(f'- [{p}]({p})')
    if rows:
        avg=sum(r['metrics']['train/iteration_seconds'] for r in rows)/len(rows)
        lines+=['',f'平均每轮耗时：{avg:.2f} 秒。详细训练曲线见 metrics.jsonl 和 W&B；所有评测逐例结果均保留。']
    if continuation:lines+=['','## 夜间续训来源','',f"父checkpoint：{continuation.get('parent_checkpoint')}；导入更新{continuation.get('update')}，原生步{continuation.get('native_steps')}。新增更新：{result['additional_updates']}；新增原生步：{result['additional_native_steps']}。权重、优化器及RNG逐项相同：{continuation.get('weights_optimizer_rng_exact')}。上一阶段预算结束的活动环境不跨进程恢复。"]
    if migration:lines+=['','## 四卡受控续训','',f'父阶段: {migration.get("parent_run")}；导入更新: {migration.get("update")}；导入原生步: {migration.get("native_steps")}；原始截止时间保持不变。',f'权重/optimizer 原样保留: {migration.get("weights_optimizer_exact")}；原三卡 RNG 前缀保留: {migration.get("old_rng_prefix_exact")}；重启丢弃 checkpoint 活动轨迹: {migration.get("discarded_checkpoint_active_count")}；丢弃尚未提交原生 trace 步: {migration.get("uncommitted_native_trace_steps")}。']
    lines+=['','## 完成轨迹与限制','',f'自然完成轨迹: {len(episodes)}；重启截断快照: {result["restart_truncated"]}；预算结束未完成: {result["budget_truncated"]}；训练初始化失败: {result["initialization_failures"]}。',str(outcomes),'','本阶段从上一轮final checkpoint恢复LoRA/critic、优化器和全部RNG，奖励权重不变。扶正零成功时不宣称已学会；不使用未经验证的 BC 示范。固定评测报告有效初态和初始化失败，语言对照单独见预检。Place 的最终状态成功判定仍允许空中松手后恰好落稳；物理预检中这种动作总回报更低、没有受控接触/释放/撤离事件奖励，不能把终态成功率当作完整动作质量。重启截断记录来自 checkpoint 活动环境快照，崩溃中的未写出步不保证精确恢复。']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return result
if __name__=='__main__':summary(sys.argv[1])
