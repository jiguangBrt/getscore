"""Own long-run utilities; base weights remain immutable."""
import os
os.environ.setdefault('RL_SKILLS_CONFIG','/data/wangyihan/getscore-rl-skills/configs/current.json')
from runtime import BASE,CFG,save,cpu_tree,build_model as build_base
import torch,math,random,numpy as np
from pathlib import Path

class LoRALinear(torch.nn.Module):
    def __init__(self,base,rank,alpha):
        super().__init__();self.base=base;self.scale=alpha/rank
        self.lora_A=torch.nn.Parameter(torch.empty(rank,base.in_features,dtype=torch.float32,device=base.weight.device))
        self.lora_B=torch.nn.Parameter(torch.zeros(base.out_features,rank,dtype=torch.float32,device=base.weight.device))
        torch.nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))
        base.requires_grad_(False)
    @property
    def weight(self):return self.base.weight
    @property
    def bias(self):return self.base.bias
    def forward(self,x):
        y=self.base(x)
        z=torch.nn.functional.linear(torch.nn.functional.linear(x.float(),self.lora_A),self.lora_B)
        return y+(z*self.scale).to(y.dtype)

class LoRAFeedForward(torch.nn.Module):
    def __init__(self,base,rank,alpha):
        super().__init__();self.base=base;self.scale=alpha/rank
        f,h=base.features,base.hidden_dim;device=base.w_gating.device
        self.lora_A_gate=torch.nn.Parameter(torch.empty(2,f,rank,device=device,dtype=torch.float32))
        self.lora_B_gate=torch.nn.Parameter(torch.zeros(2,rank,h,device=device,dtype=torch.float32))
        self.lora_A_out=torch.nn.Parameter(torch.empty(h,rank,device=device,dtype=torch.float32))
        self.lora_B_out=torch.nn.Parameter(torch.zeros(rank,f,device=device,dtype=torch.float32))
        torch.nn.init.normal_(self.lora_A_gate,std=1/math.sqrt(f));torch.nn.init.normal_(self.lora_A_out,std=1/math.sqrt(h));base.requires_grad_(False)
    def forward(self,x):
        from rlinf.models.embodiment.openpi.modules.utils import gelu_glu
        dtype=x.dtype
        gate=x@self.base.w_gating[0].to(dtype)
        up=x@self.base.w_gating[1].to(dtype)
        gate=gate+((x.float()@self.lora_A_gate[0]@self.lora_B_gate[0])*self.scale).to(dtype)
        up=up+((x.float()@self.lora_A_gate[1]@self.lora_B_gate[1])*self.scale).to(dtype)
        act=gelu_glu(gate,up)
        return act@self.base.w_linear.to(dtype)+((act.float()@self.lora_A_out@self.lora_B_out)*self.scale).to(dtype)

def build_model(device):
    model=build_base(device);model.requires_grad_(False);targets=[]
    # Expert index 1 is the action expert. Expert 0 (VLM) stays frozen.
    for name,module in list(model.named_modules()):
        if not isinstance(module,torch.nn.Linear):continue
        expert_attn=any(name.endswith(f'.attn.{q}.1') for q in ('q_proj','k_proj','v_proj','o_proj'))
        expert_mlp='.mlps.1.' in name
        if expert_attn or expert_mlp:
            parent_name,child=name.rsplit('.',1);parent=model.get_submodule(parent_name)
            parent._modules[child]=LoRALinear(module,CFG['lora_rank'],CFG['lora_alpha']);targets.append(name)
    assert len(targets)==72,targets
    for name,module in list(model.named_modules()):
        if name.endswith('.mlps.1') and hasattr(module,'w_gating'):
            parent_name,child=name.rsplit('.',1);model.get_submodule(parent_name)._modules[child]=LoRAFeedForward(module,CFG['lora_rank'],CFG['lora_alpha']);targets.append(name)
    assert len(targets)==90,targets
    model.value_head.requires_grad_(True);model.eval()
    model.lora_targets=targets
    return model

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def trainable_state(model):return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
def apply_state(model,state):
    params=dict(model.named_parameters())
    assert set(state)=={n for n,p in params.items() if p.requires_grad}
    with torch.no_grad():
        for n,v in state.items():params[n].copy_(v.to(params[n].device))
def stack_obs(rows):
    return {k:(sum((r[k] for r in rows),[]) if isinstance(rows[0][k],list) else np.concatenate([r[k] for r in rows],axis=0)) for k in rows[0]}
def slice_tree(x,i):
    if torch.is_tensor(x):return x[i:i+1].detach().cpu().clone()
    if isinstance(x,np.ndarray):return x[i:i+1].copy()
    if isinstance(x,dict):return {k:slice_tree(v,i) for k,v in x.items()}
    if isinstance(x,list):return x[i:i+1]
    raise TypeError(type(x))
def batch_tree(rows,device):
    def cat(xs):
        x=xs[0]
        if torch.is_tensor(x):return torch.cat(xs,0).to(device)
        if isinstance(x,np.ndarray):return np.concatenate(xs,0)
        if isinstance(x,list):return sum(xs,[])
        if isinstance(x,dict):return {k:cat([v[k] for v in xs]) for k in x}
        raise TypeError(type(x))
    return cat(rows)
def atomic_json(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');save(q,data);q.replace(p)
def atomic_torch(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');torch.save(data,q);q.replace(p)
