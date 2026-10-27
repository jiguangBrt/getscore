"""Kimi Code ACP adapter using the same RPent toolkit and artifacts as Codex."""
from __future__ import annotations

import json
from pathlib import Path
import queue
import subprocess
import threading
import time
from typing import Any

from rpent.planner.base import Planner, PlannerResult
from rpent.planner.utils.http_mcp_server import HttpMcpServer


class KimiPlanner(Planner):
    def open_session(self, request, *, cwd, mcp_servers):
        return request('session/new',dict(cwd=cwd,mcpServers=mcp_servers),45)

    def __init__(self, *, output_dir, planner_timeout_s=None, model=None, reasoning_effort='high', **kwargs):
        self.output=Path(output_dir)
        self.timeout=planner_timeout_s or 1200
        self.model=model or 'kimi-code/k3.1-pro'
        self.effort=reasoning_effort

    def solve(self, *, system_prompt, user_message, toolkit, max_turns, **kwargs):
        prompt=system_prompt+'\n\n'+user_message
        (self.output/'kimi_prompt.txt').write_text(prompt)
        messages=[];finish=None;error=None;counter=0;tool_calls=0;sid=None;native_done=False
        original_execute=toolkit.execute_tool
        started=time.monotonic();deadline=started+self.timeout
        server=HttpMcpServer(toolkit)
        url=server.start()
        events=queue.Queue()
        raw_file=(self.output/'kimi_acp.stream.jsonl').open('w')
        tool_file=(self.output/'kimi_tool_calls.jsonl').open('w')
        text_file=(self.output/'kimi.txt').open('w')
        write_lock=threading.Lock()
        proc=None;reader=None

        def capture_tool(name, arguments):
            nonlocal finish,tool_calls,native_done
            tool_started=time.time()
            result=original_execute(name,arguments)
            tool_calls+=1
            record=dict(time_unix=time.time(),started_at_unix=tool_started,tool=name,arguments=arguments,
                result=result.data,images=len(result.images),is_error=result.is_error)
            with write_lock:
                tool_file.write(json.dumps(record,default=str)+'\n');tool_file.flush()
            env=getattr(getattr(toolkit,'primitives',None),'env',None)
            if toolkit.solved() or bool(getattr(env,'truncated',False)):
                native_done=True
            if name=='finish' and not result.is_error:
                finish={'status':arguments['status'],'summary':arguments['summary']}
            return result

        def send(value):
            with write_lock:
                raw_file.write(json.dumps({'direction':'client','time_unix':time.time(),'message':value})+'\n');raw_file.flush()
                proc.stdin.write(json.dumps(value)+'\n');proc.stdin.flush()

        def consume():
            try:
                for line in proc.stdout:
                    value=json.loads(line)
                    with write_lock:
                        raw_file.write(json.dumps({'direction':'server','time_unix':time.time(),'message':value})+'\n');raw_file.flush()
                    events.put(value)
            except Exception as exc:events.put({'reader_error':repr(exc)})
            finally:events.put({'eof':True})

        def dispatch(value):
            if value.get('method')=='session/update':
                update=value.get('params',{}).get('update',{})
                if update.get('sessionUpdate')=='agent_message_chunk':
                    content=update.get('content',{})
                    if content.get('type')=='text':
                        text=content.get('text','');text_file.write(text);text_file.flush()
                        messages.append({'role':'assistant','content':text})
                return
            if 'id' in value and 'method' in value:
                if value['method']=='session/request_permission':
                    options=value.get('params',{}).get('options',[])
                    allowed=next((o for o in options if o.get('kind') in ['allow_once','allow_always']),None)
                    outcome={'outcome':'selected','optionId':allowed['optionId']} if allowed else {'outcome':'cancelled'}
                    send({'jsonrpc':'2.0','id':value['id'],'result':{'outcome':outcome}})
                else:
                    send({'jsonrpc':'2.0','id':value['id'],'error':{'code':-32601,'message':'Client capability not advertised'}})

        def request(method,params,timeout=None):
            nonlocal counter
            counter+=1;request_id=counter
            send({'jsonrpc':'2.0','id':request_id,'method':method,'params':params})
            end=min(deadline,time.monotonic()+(timeout or self.timeout))
            while time.monotonic()<end:
                if method=='session/prompt' and native_done:
                    send({'jsonrpc':'2.0','method':'session/cancel','params':{'sessionId':sid}})
                    return dict(stopReason='environment_terminal',origin='client_native_environment_guard')
                try:value=events.get(timeout=min(1,max(.01,end-time.monotonic())))
                except queue.Empty:continue
                if value.get('eof') or value.get('reader_error'):raise RuntimeError(f'Kimi ACP stream closed: {value}')
                if value.get('id')==request_id and 'method' not in value:
                    if 'error' in value:raise RuntimeError(f'Kimi ACP {method}: {value["error"]}')
                    return value['result']
                dispatch(value)
            raise TimeoutError(f'Kimi ACP {method} exceeded {self.timeout}s wall budget')

        stderr=(self.output/'kimi_stderr.log').open('w')
        toolkit.execute_tool=capture_tool
        try:
            proc=subprocess.Popen(['/home/jh/.kimi-code/bin/kimi','acp'],cwd=self.output,
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,text=True,bufsize=1)
            reader=threading.Thread(target=consume,daemon=True);reader.start()
            info=request('initialize',dict(protocolVersion=1,clientCapabilities={},clientInfo={'name':'getscore-long-benchmark','version':'1'}),30)
            session=self.open_session(request,cwd=str(self.output),mcp_servers=[dict(type='http',name='rpent',url=url,headers=[])])
            sid=session['sessionId']
            settings={}
            for key,value in [('model',self.model),('thinking',self.effort),('mode','yolo')]:
                configured=request('session/set_config_option',dict(sessionId=sid,configId=key,value=value),30)
                option=next((o for o in configured.get('configOptions',[]) if o['id']==key),None)
                if option is None or option['currentValue']!=value:
                    raise RuntimeError(f'Kimi did not confirm requested {key}={value}: {configured}')
                settings[key]=option
            (self.output/'kimi_session.json').write_text(json.dumps(dict(initialize=info,session=session,confirmed_settings=settings),indent=2)+'\n')
            response=request('session/prompt',dict(sessionId=sid,prompt=[dict(type='text',text=prompt)]))
            (self.output/'kimi_response.json').write_text(json.dumps(response,indent=2)+'\n')
            if not finish and not toolkit.solved() and not native_done:
                # CLI 2.3.1 can return end_turn for a provider 429 without an ACP error.
                # Its persisted wire identifies an LLM request that never completed.
                wire_paths=list((Path.home()/'.kimi-code/sessions').glob(f'*/{sid}/agents/main/wire.jsonl'))
                if wire_paths:
                    records=[json.loads(line) for line in wire_paths[0].read_text().splitlines()]
                    requests=[r.get('time',0) for r in records if r.get('type')=='llm.request']
                    ended=[r.get('time',0) for r in records if r.get('event',{}).get('type')=='step.end']
                    if requests and max(requests)>max(ended,default=0):
                        raise RuntimeError('Kimi provider infrastructure interruption: ACP ended with an unfinished LLM request; check provider quota/service before resuming')
                if not messages and tool_calls==0:
                    raise RuntimeError('Kimi provider infrastructure interruption: empty ACP turn with no model output or tool calls')
            if not finish:
                finish={'status':'success' if toolkit.solved() else 'failure','summary':f'Kimi ACP ended with {response.get("stopReason")}; native success={toolkit.solved()}'}
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}'
            text_file.write('\n[kimi-planner] '+error+'\n');text_file.flush()
            if proc and proc.poll() is None and sid:
                try:send({'jsonrpc':'2.0','method':'session/cancel','params':{'sessionId':sid}})
                except Exception:pass
        finally:
            toolkit.cancel_active_and_wait()
            if proc and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            if reader is not None:reader.join(timeout=5)
            toolkit.execute_tool=original_execute
            server.stop()
            for f in [raw_file,tool_file,text_file,stderr]:f.close()
        return PlannerResult(finish_result=finish,messages=messages,error=error,
            stats=dict(backend='kimi_code_acp',model=self.model,reasoning_effort=self.effort,
                context_window=1048576,elapsed_s=round(time.monotonic()-started,2),tool_calls=tool_calls,
                native_environment_terminal=native_done,
                max_turns_requested=max_turns,turn_limit_note='ACP does not expose model-turn limits; shared wall and native-action budgets enforced',
                output_path=str(self.output/'kimi.txt'),raw_stream_path=str(self.output/'kimi_acp.stream.jsonl')))
