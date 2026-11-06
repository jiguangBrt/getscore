"""User-authorized one-time resource migration at a saved complete update boundary."""
from train_runtime import BASE,torch
from pathlib import Path
import json,os,signal,time,shutil,hashlib,subprocess,datetime
PROFILE=Path(__file__).resolve().parents[1]
PARENT=BASE/'runs/pi05-skills-v3-6h-20261104-115938'
OUT=BASE/'runs/pi05-skills-v3-6h-20261104-115938-four-gpu'
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def same_tree(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same_tree(x,y) for x,y in zip(a,b))
    return a==b

def migrate():
    cfg=read(PROFILE/'configs/four-gpu.json');evidence=read(PROFILE/'acceptance.json')
    assert cfg['runtime_implementation_status']=='validated_v3' and evidence['passed']
    for relative,digest in evidence['runtime_sha256'].items():assert hashlib.sha256((BASE/relative).read_bytes()).hexdigest()==digest,relative
    assert not OUT.exists(),'refuse a second cutover or mixing output directories'
    supervisor=read(PARENT/'supervisor-pid.json')['pid'];trainer=read(PARENT/'supervisor-status.json')['pid']
    for pid in (supervisor,trainer):
        assert Path(f'/proc/{pid}').stat().st_uid==os.getuid()
        assert str(PARENT).encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
    assert os.getsid(trainer)==trainer and os.getsid(supervisor)==supervisor
    minimum=read(PARENT/'status.json')['update'];start=time.time();last_mtime=None;frozen=False
    try:
        while time.time()-start<900:
            p=PARENT/'checkpoints/latest.pt';mt=p.stat().st_mtime_ns
            if mt!=last_mtime:
                last_mtime=mt;candidate=torch.load(p,map_location='cpu',weights_only=False)
                if candidate['update']>=minimum:
                    os.kill(trainer,signal.SIGSTOP);frozen=True
                    current=torch.load(PARENT/'scratch/current-policy.pt',map_location='cpu',weights_only=False)
                    rows=(PARENT/'metrics.jsonl').read_text().splitlines();last=json.loads(rows[-1])
                    latest=torch.load(p,map_location='cpu',weights_only=False)
                    if latest['update']==current['version']==last['update'] and latest['native_steps']==last['native_steps']:
                        ck=latest;break
                    os.kill(trainer,signal.SIGCONT);frozen=False
            time.sleep(.2)
        else:raise TimeoutError('no fresh clean checkpoint boundary within15min; original run left running')
        OUT.mkdir();(OUT/'checkpoints').mkdir();(OUT/'scratch').mkdir()
        # Parent can no longer publish an update after this frozen boundary.
        source=PARENT/'checkpoints/latest.pt';source_copy=OUT/'checkpoints'/f'parent-checkpoint-u{ck["update"]}.pt';shutil.copy2(source,source_copy)
        parent_sha=hashlib.sha256(source_copy.read_bytes()).hexdigest()
        original_weights={n:v.clone() for n,v in ck['trainable_state'].items()};original_optimizer=ck['optimizer']
        old_rng_cuda=list(ck['rng']['cuda']);assert len(old_rng_cuda)==3
        ck['rng']['cuda']=old_rng_cuda+[torch.Generator(device='cuda:3').manual_seed(cfg['seed']).get_state()]
        assert all(torch.equal(a,b) for a,b in zip(old_rng_cuda,ck['rng']['cuda'][:3])) and len(ck['actor_states'])==2
        report=OUT/'preflight-acceptance.json';shutil.copy2(PROFILE/'acceptance.json',report)
        cfg['runtime_preflight_report']=str(report);write(OUT/'effective-config.json',cfg)
        old_meta=read(PARENT/'run-meta.json');meta=dict(old_meta,run_name=OUT.name,parent_run=str(PARENT),resource_migration_at=time.time())
        assert meta['started_at']==old_meta['started_at'] and meta['deadline']==old_meta['deadline']==cfg['absolute_deadline']
        write(OUT/'run-meta.json',meta)
        ck['config']=cfg;ck['run_meta']=meta;ck['source_manifest']={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (PROFILE/'code').glob('*.py')}
        ck['source_manifest']['git_commit']=subprocess.run(['git','rev-parse','HEAD'],cwd=BASE,capture_output=True,text=True,check=True).stdout.strip()
        ck['resource_migration']=dict(parent_run=str(PARENT),parent_checkpoint=str(source_copy),parent_checkpoint_sha256=parent_sha,update=ck['update'],native_steps=ck['native_steps'],old_physical_gpus=[0,1,3],new_physical_gpus=[0,1,2,3],cuda_order=[0,1,3,2],old_envs=16,new_envs=24,old_global_chunks=512,new_global_chunks=768,discarded_checkpoint_active_count=len(ck.get('active_episodes',[])),new_actor_id=2,new_actor_rng='fresh seed42; actor-id2 reset generator',deadline_unchanged=True)
        tmp=OUT/'checkpoints/latest.pt.tmp';torch.save(ck,tmp);tmp.replace(OUT/'checkpoints/latest.pt')
        reloaded=torch.load(OUT/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
        assert same_tree(original_weights,reloaded['trainable_state']) and same_tree(original_optimizer,reloaded['optimizer'])
        assert all(torch.equal(a,b) for a,b in zip(old_rng_cuda,reloaded['rng']['cuda'][:3]))
        for name in ('metrics.jsonl','episodes.jsonl','events.jsonl'):
            if (PARENT/name).exists():shutil.copy2(PARENT/name,OUT/name)
        for p in PARENT.glob('evaluation-*.json'):shutil.copy2(p,OUT/p.name)
        # Preserve initial/best/milestone artifacts; source remains authoritative and intact.
        for p in (PARENT/'checkpoints').glob('*.pt'):
            if p.name!='latest.pt':os.link(p,OUT/'checkpoints'/p.name)
        migration=dict(ck['resource_migration'],state='checkpoint_verified_before_cutover',supervisor_old=supervisor,trainer_old=trainer,parent_last_reported_update=last['update'],weights_optimizer_exact=True,old_rng_prefix_exact=True,parent_start=old_meta['started_at'],deadline=old_meta['deadline'],discarded_active=ck.get('active_episodes',[]))
        write(OUT/'MIGRATION.json',migration)
        # Stop only this user's known original supervisor/session. Persistent checkpoint is sealed.
        os.kill(supervisor,signal.SIGTERM);time.sleep(1.2)
        try:os.killpg(trainer,signal.SIGKILL)
        except ProcessLookupError:pass
        frozen=False
        for _ in range(60):
            proc=Path(f'/proc/{supervisor}/stat')
            if not proc.exists() or proc.read_text().split()[2]=='Z':break
            time.sleep(1)
        else:raise RuntimeError('old supervisor did not exit; checkpoint retained, new training not launched')
        native_observed=0
        for path in PARENT.glob('environments-actor*/env*/episode*/native-trace.jsonl'):
            with open(path,'rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):native_observed+=block.count(b'\n')
        migration.update(uncommitted_native_trace_steps=max(0,native_observed-ck['native_steps']),state='launched',main_runtime_unmodified=True)
        old_status=read(PARENT/'status.json');old_status.update(state='resource_migrated',successor=str(OUT),migration_update=ck['update'],migration_native_steps=ck['native_steps']);write(PARENT/'status.json',old_status)
        write(PARENT/'supervisor-status.json',dict(state='resource_migrated',successor=str(OUT),update=ck['update'],native_steps=ck['native_steps'],finished=time.time(),reason='user-authorized same-model four-GPU expansion'))
        from v3_summary import summary
        summary(PARENT)
        cmd=[str(PROFILE/'run.sh'),'v3_supervise.py','--out',str(OUT),'--hours','6','--envs','24','--chunks','32']
        env=dict(os.environ,RL_SKILLS_CONFIG=str(OUT/'effective-config.json'))
        with open(OUT/'supervisor.log','ab',buffering=0) as log:
            child=subprocess.Popen(cmd,cwd=BASE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        migration.update(supervisor_new=child.pid,command=cmd,launched_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());write(OUT/'MIGRATION.json',migration);write(PARENT/'SUCCESSOR.json',migration)
        with open(OUT/'events.jsonl','a') as f:f.write(json.dumps(dict(time=time.time(),kind='user_authorized_resource_migration',**ck['resource_migration']))+'\n')
        print(json.dumps(migration,indent=2),flush=True)
    finally:
        if frozen:
            try:os.kill(trainer,signal.SIGCONT)
            except ProcessLookupError:pass
if __name__=='__main__':migrate()
