"""Privileged critic context; the actor keeps the original pi05 input contract."""
import torch
from rlinf.models.embodiment.openpi.tasks.rl import Pi0RL

CONTEXT_FIELDS = ['remaining_fraction', 'hold_fraction', 'approach_credit', 'height_credit',
                  'hold_credit', 'contact_bonus_claimed', 'clearance_scaled', 'xy_scaled',
                  'bilateral_contact', 'verified_grasp', 'relative_position_drift_scaled',
                  'relative_rotation_drift_scaled', 'linear_speed_scaled', 'angular_speed_scaled',
                  'hold_height_min_scaled', 'hold_height_max_scaled', 'contact_count_fraction',
                  'descent_deadband_consumed', 'settling_credit']


class ContextValueHead(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        first = base.mlp[0]
        expanded = torch.nn.Linear(first.in_features + len(CONTEXT_FIELDS), first.out_features,
                                   device=first.weight.device, dtype=first.weight.dtype)
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, :first.in_features].copy_(first.weight)
            expanded.bias.copy_(first.bias)
        base.mlp[0] = expanded
        self.core = base
        self.context = None

    def forward(self, x):
        assert self.context is not None and len(self.context) == len(x)
        return self.core(torch.cat([x, self.context.to(x)], dim=-1))


class ContextPi0RL(Pi0RL):
    def _build_prefix_cache_for_actor(self, observation):
        cache=getattr(self,'_frozen_prefix_cache',None)
        key=getattr(self,'_frozen_prefix_key',None)
        if cache is None or key is None:
            return super()._build_prefix_cache_for_actor(observation)
        if key not in cache:
            cache[key]=super()._build_prefix_cache_for_actor(observation)
        else:
            self._frozen_prefix_hits+=1
        return cache[key]

    @torch.no_grad()
    def predict_action_batch(self, env_obs, mode='eval', **kwargs):
        if 'critic_context' in env_obs:
            context = torch.as_tensor(env_obs['critic_context'], device=self.device, dtype=torch.float32)
        else:
            assert mode=='eval','training requires the true critic context'
            # The evaluation actor never uses this head; deployment need not
            # provide simulator-only critic features.
            context=torch.zeros((len(env_obs['states']),len(CONTEXT_FIELDS)),device=self.device)
        self.value_head.context = context
        clean = {k: v for k, v in env_obs.items() if k != 'critic_context'}
        actions, result = super().predict_action_batch(clean, mode=mode, **kwargs)
        if 'forward_inputs' in result:
            result['forward_inputs']['critic_context'] = context.detach().clone()
        return actions, result

    def default_forward(self, forward_inputs, **kwargs):
        self.value_head.context = forward_inputs['critic_context']
        self._frozen_prefix_key=tuple(forward_inputs.get('frozen_prefix_key',[])) or None
        clean = {k: v for k, v in forward_inputs.items()
                 if k not in ('critic_context','frozen_prefix_key')}
        try:return super().default_forward(clean, **kwargs)
        finally:self._frozen_prefix_key=None
