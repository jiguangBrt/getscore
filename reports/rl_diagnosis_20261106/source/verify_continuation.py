from train_runtime import BASE,torch,np
from pathlib import Path
import json,hashlib,subprocess,datetime
out=BASE/'runs/pi05-skills-v3-6h-20261104-115938-four-gpu';migration=json.loads((out/'MIGRATION.json').read_text())
parent=torch.load(migration['parent_checkpoint'],map_location='cpu',weights_only=False);initial=torch.load(out/'checkpoints/migration-start.pt',map_location='cpu',weights_only=False)
def same(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b
assert initial['update']==parent['update']==95 and initial['native_steps']==parent['native_steps']==385875
assert same(parent['trainable_state'],initial['trainable_state']) and same(parent['optimizer'],initial['optimizer'])
assert same(parent['actor_states'],initial['actor_states']) and same(parent['rng']['cuda'],initial['rng']['cuda'][:3]) and len(initial['rng']['cuda'])==4
assert all(float(v['step'])==95 for v in initial['optimizer']['state'].values())
assert hashlib.sha256(Path(migration['parent_checkpoint']).read_bytes()).hexdigest()==migration['parent_checkpoint_sha256']
assert any(torch.count_nonzero(v)>0 for n,v in initial['trainable_state'].items() if 'lora_B' in n)
rows=[json.loads(l) for l in (out/'metrics.jsonl').read_text().splitlines()];first=next(r for r in rows if r['update']>95)
assert first['update']==96 and first['learner']['chunks']==768 and len(first['learner']['per_rank'])==4 and all(r['chunks']==192 for r in first['learner']['per_rank'])
assert first['learner']['parallel_gradient_gpus']==[0,1,3,2] and first['learner']['max_logprob_recompute_error']<=.001
assert not first['learner']['critic_warmup'] and first['learner']['actor_lr']==1e-6 and first['learner']['gradient_norm']>0
assert [r['physical_gpu'] for r in first['rollouts']]==[1,3,2] and all(r['chunks']==256 for r in first['rollouts'])
assert first['native_steps']==385875+sum(r['native_steps'] for r in first['rollouts'])
meta=json.loads((out/'run-meta.json').read_text());old_meta=json.loads((Path(migration['parent_run'])/'run-meta.json').read_text());assert meta['started_at']==old_meta['started_at'] and meta['deadline']==old_meta['deadline']
events=[json.loads(l) for l in (out/'events.jsonl').read_text().splitlines()];ready=next(r for r in reversed(events) if r['kind']=='ready');assert ready['actors']==3 and ready['envs']==24
pids=[migration['supervisor_new'],json.loads((out/'supervisor-status.json').read_text())['pid']]+ready['pids'];cfg=json.loads((out/'effective-config.json').read_text())
for pid in pids:
    env=Path(f'/proc/{pid}/environ').read_bytes().split(b'\0');cv=next(x.split(b'=',1)[1].decode() for x in env if x.startswith(b'CUDA_VISIBLE_DEVICES='));assert cv==','.join(cfg['gpu_uuids'])
wandb={str(p.relative_to(out)):p.stat().st_size for p in (out/'wandb').glob('offline-run-*/run-*.wandb')};assert wandb and all(v>0 for v in wandb.values())
current=torch.load(out/'scratch/current-policy.pt',map_location='cpu',weights_only=False);delta=max(float((v-initial['trainable_state'][n]).abs().max()) for n,v in current['trainable_state'].items());assert delta>0
proof=dict(passed=True,verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_parent_sha256=migration['parent_checkpoint_sha256'],migration_update=95,migration_native_steps=385875,optimizer_state_steps_imported=95,weights_optimizer_actor_rng_exact=True,cuda_rng_old_prefix_exact=True,added_cuda_rng_seed=42,first_update=dict(update=first['update'],native_steps=first['native_steps'],increment_native_steps=first['native_steps']-385875,chunks=768,rank_chunks=[r['chunks'] for r in first['learner']['per_rank']],physical_order=[0,1,3,2],sampler_physical_gpus=[1,3,2],logprob_max_error=first['learner']['max_logprob_recompute_error'],actor_lr=first['learner']['actor_lr'],critic_warmup=False,approx_kl=first['learner']['approx_kl']),current_synced_policy=dict(version=current['version'],max_weight_change_since_migration=delta),supervisor_pid=pids[0],trainer_pid=pids[1],actor_pids=ready['pids'],uuid_environments_verified=True,deadline=meta['deadline'],original_started_at=meta['started_at'],discarded_checkpoint_episodes=15,uncommitted_native_trace_steps=2810,wandb_files=wandb,gpu_query=subprocess.run(['nvidia-smi','-i','0,1,2,3','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv'],capture_output=True,text=True,check=True).stdout)
(out/'continuation-verification.json').write_text(json.dumps(proof,indent=2)+'\n');(BASE/'reports/v3-four-gpu-continuation-verification-20261104.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))
