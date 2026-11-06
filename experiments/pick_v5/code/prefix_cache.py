"""Update-local reuse of immutable, frozen VLM prefix outputs.

Each key identifies the complete original eight-row sampling group in order.
The action expert and value head are always recomputed with current parameters.
The cache is discarded before the next rollout or model update call.
"""
from contextlib import contextmanager


@contextmanager
def frozen_prefix_scope(model,rows,enabled):
    if not enabled:
        yield
        return
    assert model.rl_cfg.train_expert_only
    # Only action-expert adapters and critic may be trainable in this profile.
    def allowed(n):
        return n.startswith('value_head.') or '.mlps.1.lora_' in n or any(
            f'.attn.{q}.1.lora_' in n for q in ('q_proj','k_proj','v_proj','o_proj'))
    assert all(not p.requires_grad or allowed(n) for n,p in model.named_parameters())
    assert getattr(model,'_frozen_prefix_cache',None) is None
    for r in rows:
        assert 'frozen_prefix_key' not in r['forward_inputs']
        r['forward_inputs']['frozen_prefix_key']=[f"{r['sampling_group']}:{r['sampling_index']}"]
    model._frozen_prefix_cache={};model._frozen_prefix_hits=0
    try:yield
    finally:
        model._last_prefix_cache_stats=dict(entries=len(model._frozen_prefix_cache),hits=model._frozen_prefix_hits)
        model._frozen_prefix_cache=None;model._frozen_prefix_key=None
        for r in rows:r['forward_inputs'].pop('frozen_prefix_key',None)
