"""Exact conditional Gaussian drift from the strong initialization, by object."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from train_runtime import *
from prefix_cache import frozen_prefix_scope
from rlinf.models.embodiment.openpi.sampling import rl_sampler


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True);p.add_argument('--current',required=True)
    p.add_argument('--reference',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    seed_all(42);torch.cuda.set_device(0);model=build_model('cuda:0')
    data=torch.load(a.batch,map_location='cpu',weights_only=False);groups={}
    for row in data:groups.setdefault(row['sampling_group'],[]).append(row)
    ordered=[r for group in groups.values() for r in sorted(group,key=lambda r:r['sampling_index'])]
    captured=[]
    handle=model.action_out_proj.register_forward_hook(lambda module,inputs,output:captured.append(output.detach()))
    outputs={};record={};errors=[]
    with frozen_prefix_scope(model,ordered,True),torch.no_grad():
        for label,path in [('current',a.current),('reference',a.reference)]:
            state=torch.load(path,map_location='cpu',weights_only=False);apply_state(model,state['trainable_state'])
            outputs[label]=[]
            for off in range(0,len(ordered),8):
                rows=ordered[off:off+8];fi=batch_tree([r['forward_inputs'] for r in rows],'cuda:0');captured.clear()
                result=model(forward_inputs=fi);assert len(captured)==1
                idx=fi['denoise_inds'][:,0];x=fi['chains'][torch.arange(8,device='cuda:0'),idx]
                mu,std=rl_sampler.sample_mean_var(x.float(),captured[0].float(),idx,noise_method=CFG['noise_method'],noise_level=CFG['noise_level'],num_steps=CFG['flow_steps'])
                outputs[label].append((mu.cpu(),std.cpu()))
                if label=='reference':
                    old=torch.cat([r['old_logprobs'] for r in rows],0).to('cuda:0');errors.append(float((result['logprobs']-old).abs().max()))
            record[label]=dict(path=path,update=state['update'])
    handle.remove();by=defaultdict(list)
    for j,((cur,std),(ref,refstd)) in enumerate(zip(outputs['current'],outputs['reference'])):
        assert torch.equal(std,refstd)
        rows=ordered[j*8:j*8+8]
        mask=torch.as_tensor(np.stack([r['valid_action_mask'] for r in rows]))[:,:,None]
        kl=(((cur-ref)**2/(2*std.square().clamp_min(1e-12)))[:,:8,:7]*mask).sum((1,2))
        for row,v in zip(rows,kl):by[row['object_type']].append(float(v))
    result=dict(source=record,reference_logprob_error=max(errors),
        mean_kl=float(np.mean([v for values in by.values() for v in values])),
        by_object={t:dict(chunks=len(v),mean_kl=float(np.mean(v)),max_kl=max(v)) for t,v in by.items()},
        role='offline parameter-drift diagnosis on recorded states; no training or success episodes')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
