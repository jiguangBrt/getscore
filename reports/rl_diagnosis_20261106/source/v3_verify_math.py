from train_runtime import *
from ppo_update import compute_gradients
from v3_train import finish_gae,milestone_due
import argparse,time,json

def verify(out):
    out=Path(out);start=time.time();checks={}
    assert not milestone_due(7199,0,7200) and milestone_due(7200,0,7200) and not milestone_due(7201,7200,7200) and milestone_due(14400,7201,7200) and milestone_due(21600,14400,7200)
    checks['2h_4h_6h_milestones']=True
    for terminal in (False,True):
        row=dict(native_steps=3,terminated=terminal,reward=.2,old_value=.4)
        finish_gae([row],.7);expect=.2+CFG['gamma_per_native_step']**3*(0 if terminal else .7)
        assert abs(row['return']-expect)<1e-12
    checks['terminal_zero_timeout_final_observation_bootstrap']=True
    model=build_model('cuda:0');ck=torch.load(out/'checkpoints/initial.pt',map_location='cpu',weights_only=False);apply_state(model,ck['trainable_state'])
    rows=torch.load(out/'verification-batch.pt',map_location='cpu',weights_only=False);groups={}
    for r in rows:
        r['version']=3;groups.setdefault(r['sampling_group'],[]).append(r)
        assert np.array_equal(r['valid_action_mask'],np.arange(8)<r['native_steps'])
    groups=[sorted(g,key=lambda r:r['sampling_index']) for g in groups.values()]
    assert all(len(g)==8 and [r['sampling_index'] for r in g]==list(range(8)) for g in groups)
    rows=[r for g in groups for r in g];adv=np.asarray([r['advantage'] for r in rows]);kwargs=dict(update=3,microbatch=8,global_count=len(rows),adv_mean=float(adv.mean()),adv_std=float(adv.std()),device='cuda:0')
    metrics=compute_gradients(model,rows,**kwargs)
    reference={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad and p.grad is not None};reduced={n:torch.zeros_like(g) for n,g in reference.items()}
    for rank in range(4):
        part=[r for g in groups[rank::4] for r in g];compute_gradients(model,part,**kwargs)
        for n,p in model.named_parameters():
            if n in reduced and p.grad is not None:reduced[n].add_(p.grad.detach().cpu())
    diff=max(float((reference[n]-reduced[n]).abs().max()) for n in reference);scale=max(float(g.abs().max()) for g in reference.values())
    assert diff<1e-4 and diff/max(scale,1e-9)<5e-5,(diff,scale)
    actor_nonzero=sum(bool(g.abs().max()>0) for n,g in reference.items() if not n.startswith('value_head.'));assert actor_nonzero>0
    checks['four_rank_gradient_vs_serial']=dict(max_abs=diff,relative=diff/max(scale,1e-9),nonzero_actor_gradient_tensors=actor_nonzero)
    final=torch.load(out/'checkpoints/final.pt',map_location='cpu',weights_only=False);apply_state(model,final['trainable_state'])
    assert all(torch.equal(p.detach().cpu(),final['trainable_state'][n]) for n,p in model.named_parameters() if p.requires_grad)
    delta={group:max(float((v-ck['trainable_state'][n]).abs().max()) for n,v in final['trainable_state'].items() if n.startswith('value_head.')==(group=='critic')) for group in ('actor','critic')}
    assert all(v>0 for v in delta.values())
    opt=torch.optim.AdamW([dict(params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('value_head.')]),dict(params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('value_head.')])]);opt.load_state_dict(final['optimizer'])
    assert len(opt.state)==len(final['optimizer']['state']) and all(float(state['step'])==4 for state in opt.state.values())
    from v3_train import restore_rng,rng_state
    restore_rng(final['rng']);draw1=torch.randn(20);restore_rng(final['rng']);draw2=torch.randn(20);assert torch.equal(draw1,draw2)
    assert len(final['actor_states'])==3 and final['environment_resume'].startswith('unfinished episodes discarded')
    checks['checkpoint_exact_weights_optimizer_rng_reload']=dict(delta=delta,optimizer_states=len(opt.state),actor_rngs_saved=True)
    summaries=[json.loads(l) for l in (out/'metrics.jsonl').read_text().splitlines()]
    continuation=[x for row in summaries[1:] for rollout in row['rollouts'] for x in rollout['continued']]
    assert continuation and all(x['native_step']>0 for x in continuation)
    audit=json.loads((out/'evaluation-continuity-check.json').read_text());assert audit['training_state_preserved'] and all(x['training_before']==x['training_after'] for x in audit['environment_checks'])
    checks['live_episode_continues_across_updates_and_independent_eval']=dict(continued=len(continuation),evaluation_state_checks=3)
    traces=list(out.glob('environments-*/env*/episode*/native-trace.jsonl'));timeouts=[]
    for path in traces:
        trace=[json.loads(l) for l in path.read_text().splitlines()]
        assert [t['native_step'] for t in trace]==list(range(1,len(trace)+1)) and len(trace)<=200
        if trace[-1]['truncated']:assert len(trace)==200;timeouts.append(str(path))
    assert timeouts
    checks['native_step_trace_continuity_and_200_step_timeouts']=dict(traces=len(traces),timeouts=len(timeouts))
    result=dict(passed=True,checks=checks,rows=len(rows),partial_chunks=sum(r['native_steps']<8 for r in rows),logprob_max_error=metrics['max_logprob_recompute_error'],seconds=time.time()-start)
    save(out/'math-verification.json',result);print(result,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();verify(a.out)
