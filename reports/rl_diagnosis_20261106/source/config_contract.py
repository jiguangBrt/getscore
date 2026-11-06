"""Validate task specifications and reject launching a mismatched training runtime."""
from pathlib import Path
import argparse,json,math,os,sys
BASE=Path('/data/wangyihan/getscore-rl-skills')
DEFAULT_CONFIG=BASE/'configs/current.json'

def config_path():
    return Path(os.environ.get('RL_SKILLS_CONFIG',str(DEFAULT_CONFIG))).expanduser().resolve(strict=True)

def validate(config):
    errors=[]
    def need(test,message):
        if not test:errors.append(message)
    def close(a,b):return isinstance(a,(float,int)) and math.isclose(a,b,rel_tol=0,abs_tol=1e-9)
    if config.get('task_version')=='rpent-skills-v3-20261104':
        need(config.get('control_hz')==20,'v3 control_hz must be20')
        need(config.get('max_native_steps_per_skill')==200,'v3 episode budget must be200 native steps')
        need(config.get('episode_seconds')==10,'v3 duration must be10seconds')
        need(config.get('hold_time_included_in_episode') is True,'hold/stability must be included in200 steps')
        p=config.get('pick',{});q=config.get('place',{});u=config.get('upright',{})
        need(close(p.get('lift_reward_start_m'),.05),'pick reward starts at5cm')
        need(close(p.get('lift_reward_max_m'),.10),'pick height reward saturates at10cm')
        need(p.get('descent_penalty_enabled') is True,'pick descent must reduce reward')
        need(close(p.get('horizontal_free_radius_m'),0),'pick has no horizontal free region')
        need(close(p.get('horizontal_failure_radius_m'),.05),'pick horizontal failure threshold must be5cm')
        need(p.get('horizontal_failure_comparison')=='>=','5cm boundary itself must fail')
        need(close(p.get('hold_seconds'),2),'pick hold must be2seconds')
        need(close(q.get('initial_object_clearance_m'),.10),'place must start at10cm object clearance')
        need(q.get('requires_real_grasp_at_reset') is True,'place must start with a real grasp')
        need(q.get('initialization_actions_in_training_data') is False,'place setup is not on-policy data')
        need(q.get('initialization_in_200_step_budget') is False,'place setup does not consume policy budget')
        need(q.get('automatically_select_scene_plate_or_caddy') is False,'do not reuse original scene destination')
        need(close(q.get('stable_seconds'),4) and close(u.get('stable_seconds'),4),'stable observation remains4seconds')
        need(q.get('pick_descent_penalty_applies') is False and u.get('pick_descent_penalty_applies') is False,'pick-only descent penalty cannot apply to other skills')
        need(config.get('continue_env_across_updates') is True and config.get('reset_at_rollout_boundary') is False,'unfinished episodes must continue across sampling batches')
        need(config.get('preserve_training_env_during_evaluation') is True,'evaluation cannot overwrite active training environments')
        need(config.get('success',{}).get('pick_hold_steps')==40,'pick hold40steps')
        need(config.get('success',{}).get('stable_steps')==80,'place/upright stability80steps')
        need(config.get('resume_previous_run') is True and config.get('continuation_same_task') is True, 'overnight must explicitly resume the same v3 task')
        need(config.get('execute_horizon')==8,'execution chunk remains8 native actions')
        need(set(config.get('physical_gpus',[]))=={0,1,2,3},'only physical GPUs0,1,2,3 authorized in four-GPU profile')
        need(config.get('cuda_physical_order')==[0,1,3,2] and config.get('rollout_physical_gpus')==[1,3,2], 'preserve original CUDA/RNG order and append GPU2')
        need(config.get('parallel_envs')==24 and config.get('envs_per_actor')==8 and config.get('micro_batch_size')==8, 'four GPU profile must keep batch8 groups and24 environments')
        need(config.get('global_batch_chunks')==768, '24env x32 chunks must be768')
    if errors:raise ValueError('Invalid task configuration:\n- '+'\n- '.join(errors))
    return config

def guard_runtime(supported_prefix):
    path=config_path();config=validate(json.loads(path.read_text()))
    if not config.get('task_version','').startswith(supported_prefix):
        raise RuntimeError(f'Config {path} requests {config.get("task_version")}, but this entry supports {supported_prefix} only. '
                           f'v3 environment/collector implementation and preflight are pending. Read {BASE}/NEXT_TRAINING_HANDOFF.md. '
                           'Do not bypass this check or launch v2 while labeling it v3. For an intentional historical v2 reproduction, explicitly set RL_SKILLS_CONFIG to configs/long12.json.')
    return config

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path);args=parser.parse_args()
    path=args.config.resolve(strict=True) if args.config else config_path();c=validate(json.loads(path.read_text()))
    print(json.dumps(dict(structural_validation='passed',config=str(path),task_version=c.get('task_version'),episode_steps=c.get('max_native_steps_per_skill'),runtime_implementation_status=c.get('runtime_implementation_status','legacy_v2'),note='Configuration validation is not a runtime or training preflight.'),indent=2))
