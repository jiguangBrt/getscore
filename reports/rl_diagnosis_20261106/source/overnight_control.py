"""Validate an explicitly authorized, immutable same-v3 overnight continuation."""
from pathlib import Path
import hashlib,json

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_launch(cfg,meta,out,args):
    out=Path(out)
    proof=json.loads((out/'launch-validation.json').read_text())
    assert proof['passed'] and proof['weights_optimizer_rng_exact']
    assert args.envs==24 and args.chunks==32
    assert cfg['resume_previous_run'] and cfg['continuation_same_task']
    assert cfg['checkpoint_interval_seconds']==cfg['evaluation_interval_seconds']==10800
    assert meta['deadline']==cfg['absolute_deadline']<meta['reporting_deadline']==cfg['reporting_deadline']
    assert meta['reporting_deadline']==1793840400, 'hard deadline must be 2026-11-05 09:00 Asia/Shanghai'
    assert cfg['wandb_mode']=='offline'
    assert digest(out/'effective-config.json')==proof['config_sha256']
    assert digest(cfg['continuation_checkpoint'])==proof['parent_checkpoint_sha256']
    for path,sha in proof['runtime_sha256'].items():assert digest(path)==sha,f'validated source changed: {path}'
