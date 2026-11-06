from pathlib import Path
import os,sys,json,logging
BASE=Path('/data/wangyihan/getscore-rl-skills')
for path in [BASE/'src/RLinf', BASE/'src/openpi/src', BASE/'src/openpi/packages/openpi-client/src', BASE/'src/LIBERO-PRO']:
    sys.path.insert(0,str(path))
os.environ.update({'MUJOCO_GL':'egl','PYOPENGL_PLATFORM':'egl','MUJOCO_EGL_DEVICE_ID':'3','LIBERO_CONFIG_PATH':str(BASE/'configs/libero'),'LIBERO_PRO_CONFIG_PATH':str(BASE/'configs/libero'),'LIBERO_PRO_ASSET_PATH':str(BASE/'assets'),'LIBERO_ASSET_PATH':str(BASE/'assets'),'JAX_PLATFORMS':'cpu','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENPI_DATA_HOME':str(BASE/'cache/openpi'),'HF_HOME':str(BASE/'cache/huggingface'),'PYTHONNOUSERSITE':'1','TOKENIZERS_PARALLELISM':'false'})
os.environ.update({'MPLCONFIGDIR':str(BASE/'cache/matplotlib'),'NUMBA_CACHE_DIR':str(BASE/'cache/numba'),'NUMBA_NUM_THREADS':'2','TRITON_CACHE_DIR':str(BASE/'cache/triton'),'XDG_CACHE_HOME':str(BASE/'cache/xdg'),'__GL_SHADER_DISK_CACHE_PATH':str(BASE/'cache/nvidia-shader')})
CFG=json.loads((Path(os.environ.get('RL_SKILLS_CONFIG',str(BASE/'configs/smoke.json')))).read_text())
logging.basicConfig(level=logging.INFO)
def save(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def cpu_tree(x):
    import torch
    if torch.is_tensor(x):return x.detach().cpu()
    if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
    return x

def build_model(device):
    import torch
    from safetensors.torch import load_file
    from rlinf.models.embodiment.openpi.pi0_config import Pi0Config
    from rlinf.models.embodiment.openpi.tasks.rl import Pi0RL,Pi0RLConfig
    from rlinf.models.embodiment.openpi.modules.utils import set_torch_compile
    from openpi import transforms as t
    from openpi.models import model as om
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.policies.libero_policy import LiberoInputs,LiberoOutputs
    from openpi.shared.normalize import load
    set_torch_compile(False)
    torch.manual_seed(CFG['seed'])
    config=Pi0Config(pi05=True,action_horizon=CFG['action_horizon'],action_dim=32,max_token_len=200,discrete_state_input=False)
    model=Pi0RL(config,num_steps=CFG['flow_steps'],action_chunk=CFG['execute_horizon'],action_env_dim=7,config_name='pi05_libero',rl_cfg=Pi0RLConfig(add_value_head=True,noise_method=CFG['noise_method'],noise_level=CFG['noise_level'],joint_logprob=False,value_after_vlm=True,detach_critic_input=True,train_expert_only=True,config_name='pi05_libero'))
    ckpt=BASE/'checkpoints/pi05_libero_torch/model.safetensors'
    loaded=model.load_state_dict(load_file(str(ckpt)),strict=False)
    assert not loaded.unexpected_keys,loaded.unexpected_keys
    assert all(k.startswith('value_head.') for k in loaded.missing_keys),loaded.missing_keys
    model.freeze_vlm()
    # Plumbing-only update: limit changed weights to the real action output head and critic.
    for name,param in model.named_parameters():
        param.requires_grad_(name.startswith(('action_out_proj.','value_head.')))
    model.to_bfloat16_for_selected_params('bfloat16')
    norm=load(BASE/'checkpoints/pi05_libero_torch/assets/physical-intelligence/libero')
    model.setup_transforms([LiberoInputs(om.ModelType.PI05),t.Normalize(norm,use_quantiles=True),t.ResizeImages(224,224),t.TokenizePrompt(PaligemmaTokenizer(200),discrete_state_input=False),t.PadStatesAndActions(32)],[t.Unnormalize(norm,use_quantiles=True),LiberoOutputs()])
    model=model.to(device)
    model.eval()
    return model

# EGL enumeration is a different namespace from physical nvidia-smi indices.
# Query it against the exact three allowed UUIDs before importing any simulator.
if os.environ.get('CUDA_VISIBLE_DEVICES'):
    from rlinf.scheduler.hardware.accelerators.nvidia_gpu import _query_egl_index_by_cuda_ordinal
    from rlinf.utils.robosuite_compat import install_robosuite_egl_device_shim
    install_robosuite_egl_device_shim()
    mapping=_query_egl_index_by_cuda_ordinal()
    assert set(mapping)=={0,1,2,3},mapping
    assert os.environ['CUDA_VISIBLE_DEVICES']==','.join(CFG['gpu_uuids']), 'four-GPU UUID order mismatch'
    os.environ['MUJOCO_EGL_DEVICE_ID']=str(mapping[int(os.environ.get('RENDER_LOGICAL_CUDA','2'))])
    os.environ['EGL_DEVICE_ID']=str(mapping[int(os.environ.get('RENDER_LOGICAL_CUDA','2'))])
    save(BASE/'records'/f'gpu-binding-{os.getpid()}.json',dict(pid=os.getpid(),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],cuda_to_egl=mapping,render_logical_cuda=int(os.environ.get('RENDER_LOGICAL_CUDA','2')),render_egl_index=mapping[int(os.environ.get('RENDER_LOGICAL_CUDA','2'))]))
