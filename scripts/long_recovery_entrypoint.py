"""Use archived Kimi context and verified replay, with a separate suffix trace."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'RPent')]
from scripts import long_benchmark_entrypoint, long_benchmark_kimi
from robots.libero import robot_spec


class RecoveredKimiPlanner(long_benchmark_kimi.KimiPlanner):
    def solve(self,*,system_prompt,user_message,toolkit,max_turns,**kwargs):
        self.recovery=toolkit.primitives.env._client.call('env.recovery_info')
        if self.recovery['mode']=='resume':
            source=Path(os.environ['LONG_RECOVERY_SOURCE'])
            self.original_session=json.loads((source/'kimi_session.json').read_text())['session']['sessionId']
            self._finish_pending_actions(toolkit,source)
            steps=toolkit.primitives.env._benchmark_native_steps
            system_prompt=(f'Continue the SAME task and physical episode from your restored session context. '
                f'Historical replay matched the archived observations. The old 1000-action budget was increased to 2000 total; '
                f'{steps} actions have now executed. Do not reset. The prior finish(failure) was the old budget boundary. '
                f'All tools now address the restored live environment. Current state history starts at local step 0 '
                f'(the old action-1000 boundary); use view_env_state(step=-1), not stale old step indices. '
                f'Output/audit/recipe paths from the old prompt have moved: write all new output in {self.output}. '
                'Previous images and messages are historical context. Inspect the latest live images before acting. '
                'Use the same policy vocabulary, perception rules and native success criterion. '
                'Do not read other experiment outputs, simulator state, benchmark task definitions or initial-state files. '
                'Continue in place toward task success with bounded calls and corrections; finish only on native success, '
                'budget exhaustion, or a physically unrecoverable situation.')
            user_message='Resume the unfinished task with the additional native-action budget.'
        result=super().solve(system_prompt=system_prompt,user_message=user_message,toolkit=toolkit,max_turns=max_turns,**kwargs)
        result.stats.update(recovery_mode=self.recovery['mode'],resumed_from_steps=self.recovery['resumed_steps'])
        return result

    def open_session(self,request,*,cwd,mcp_servers):
        if self.recovery['mode']!='resume':
            return super().open_session(request,cwd=cwd,mcp_servers=mcp_servers)
        fork=subprocess.run(['/home/jh/.kimi-code/bin/kimi','fork',self.original_session,'--yes'],
            cwd=cwd,capture_output=True,text=True,timeout=45)
        match=re.search(r'Forked to (session_[A-Za-z0-9-]+)',fork.stdout)
        if fork.returncode or not match:
            raise RuntimeError('RECOVERY_SESSION_UNAVAILABLE: '+fork.stderr[-1000:])
        sid=match.group(1)
        (self.output/'kimi_context_recovery.json').write_text(json.dumps(dict(
            original_session_id=self.original_session,fork_session_id=sid,original_session_modified=False),indent=2)+'\n')
        try:
            session=request('session/load',dict(sessionId=sid,cwd=cwd,mcpServers=mcp_servers),45)
        except Exception as exc:
            raise RuntimeError('RECOVERY_SESSION_UNAVAILABLE: '+str(exc)) from exc
        return {**session,'sessionId':sid}

    def _finish_pending_actions(self,toolkit,source):
        import numpy as np
        episode=next(source.glob('trajectory/raw/libero/*/task_*/seed_*/episode_*'))
        with np.load(episode/'transitions.npz') as data:
            proposal=int(data['vla_chunk_id'][-1]);index=int(data['proposal_index'][-1])
        if proposal<0:return
        with np.load(episode/'proposals.npz') as data:
            original_chunk=data['actions'][proposal].copy()
            pending=original_chunk[index+1:];instruction=str(data['instruction'][proposal])
        if not len(pending):return
        p=toolkit.primitives;writer=p._flywheel;started=time.monotonic()
        p.begin_primitive('resume_pending_policy_actions')
        try:
            vla_id=writer.add_proposal(instruction,original_chunk)
            for i,action in enumerate(pending):
                obs,reward,term,trunc,_=p.env.step(action)
                writer.add_transition(action,obs,reward,term,trunc,vla_id=vla_id,proposal_index=index+1+i)
                p.set_obs(obs);p.record_frame(obs)
                if term or trunc:break
        finally:p.end_primitive()
        toolkit.get_env_state(command=dict(action='resume_pending_policy_actions',instruction=instruction),
            result=dict(executed_steps=i+1,source='unexecuted suffix of archived proposal; no new model inference'),
            elapsed_s=time.monotonic()-started)


def main():
    original_daemon=robot_spec.ProcessDaemon
    class RecoveryDaemon(original_daemon):
        def __init__(self,*,cmd,**kwargs):
            if kwargs.get('name')=='env_server':
                cmd=[cmd[0],str(ROOT/'scripts/long_recovery_env.py'),*cmd[2:]]
            super().__init__(cmd=cmd,**kwargs)
    robot_spec.ProcessDaemon=RecoveryDaemon
    long_benchmark_kimi.KimiPlanner=RecoveredKimiPlanner
    return long_benchmark_entrypoint.main()


if __name__=='__main__':sys.exit(main())
