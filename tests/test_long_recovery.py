"""Regression for completing an archived partial policy proposal."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'RPent')]
from robots.libero.flywheel import LIBERO_SPEC
from rpent.flywheel.episode import EpisodeWriter, validate_episode
from scripts.long_recovery_entrypoint import RecoveredKimiPlanner


def test_kimi_stops_at_native_action_cap_without_waiting_for_provider_finish(tmp_path,monkeypatch):
    import json
    import queue
    import threading
    from scripts import long_benchmark_kimi as kimi
    events=queue.Queue()
    env=SimpleNamespace(truncated=False)
    def execute(name,args):
        env.truncated=True
        return SimpleNamespace(data={'truncated':True},images=[],is_error=False)
    toolkit=SimpleNamespace(primitives=SimpleNamespace(env=env),execute_tool=execute,
        solved=lambda:False,cancel_active_and_wait=lambda:None)
    class Stream:
        def __iter__(self):
            while True:
                value=events.get()
                if value is None:return
                yield json.dumps(value)+'\n'
    class Process:
        def __init__(self,*args,**kwargs):
            self.stdin=SimpleNamespace(write=self.write,flush=lambda:None)
            self.stdout=Stream();self.active=True
        def write(self,line):
            msg=json.loads(line);method=msg['method']
            if method=='session/prompt':
                threading.Timer(.01,lambda:toolkit.execute_tool('move_to',{})).start()
                return  # Provider deliberately never sends a final response.
            if 'id' not in msg:return
            result={}
            if method=='session/new':result={'sessionId':'test-native-cap'}
            if method=='session/set_config_option':
                p=msg['params'];result={'configOptions':[{'id':p['configId'],'currentValue':p['value']}]}
            events.put({'jsonrpc':'2.0','id':msg['id'],'result':result})
        def poll(self):return None if self.active else 0
        def terminate(self):self.active=False;events.put(None)
        def wait(self,timeout=None):return 0
    monkeypatch.setattr(kimi.subprocess,'Popen',Process)
    monkeypatch.setattr(kimi,'HttpMcpServer',lambda toolkit:SimpleNamespace(start=lambda:'http://unused',stop=lambda:None))
    planner=kimi.KimiPlanner(output_dir=tmp_path,planner_timeout_s=2)
    result=planner.solve(system_prompt='task',user_message='run',toolkit=toolkit,max_turns=5)
    assert result.error is None
    assert result.finish_result['status']=='failure'
    assert result.stats['native_environment_terminal'] is True
    response=json.loads((tmp_path/'kimi_response.json').read_text())
    assert response['stopReason']=='environment_terminal'
    assert response['origin']=='client_native_environment_guard'


def test_partial_proposal_preserves_horizon_and_only_executes_unfinished_actions(tmp_path):
    source=tmp_path/'source'
    episode=source/'trajectory/raw/libero/libero_10/task_04/seed_000/episode_test'
    episode.mkdir(parents=True)
    chunk=np.arange(35,dtype=np.float32).reshape(5,7)/100
    np.savez(episode/'transitions.npz',vla_chunk_id=np.array([0]),proposal_index=np.array([1]))
    np.savez(episode/'proposals.npz',actions=chunk[None],instruction=np.array(['move mug']))
    obs={'main_images':np.zeros((256,256,3),np.uint8),'wrist_images':np.zeros((256,256,3),np.uint8),'states':np.zeros(8,np.float32)}
    writer=EpisodeWriter(tmp_path/'new',metadata={},spec=LIBERO_SPEC,initial_observation=obs)
    executed=[]
    class Env:
        def step(self,action):
            executed.append(action.copy())
            return obs,0,False,False,{}
    primitives=SimpleNamespace(env=Env(),_flywheel=writer,begin_primitive=writer.begin_primitive,
        end_primitive=writer.end_primitive,set_obs=lambda obs:None,record_frame=lambda obs:None)
    toolkit=SimpleNamespace(primitives=primitives,get_env_state=lambda **kwargs:None)
    planner=RecoveredKimiPlanner.__new__(RecoveredKimiPlanner)
    planner._finish_pending_actions(toolkit,source)
    np.testing.assert_array_equal(executed,chunk[2:])
    # A new prediction must remain stackable with the recovered full proposal.
    writer.begin_primitive('pi0_doubled')
    next_id=writer.add_proposal('next move',np.zeros((5,7),np.float32))
    writer.add_transition(np.zeros(7,np.float32),obs,0,False,False,vla_id=next_id,proposal_index=0)
    writer.end_primitive()
    path=writer.finalize()
    assert validate_episode(path,spec=LIBERO_SPEC)['step_count']==4
    with np.load(path/'transitions.npz') as data:
        assert data['proposal_index'].tolist()==[2,3,4,0]
    with np.load(path/'proposals.npz') as data:
        assert data['actions'].shape==(2,5,7)
