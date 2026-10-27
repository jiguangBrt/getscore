"""Wait for the supervisor to exit, then wake its original Codex app thread.

Uses the installed codex-app-tools MCP server over its normal stdio transport.
No model polling, no lifecycle hook, and no new conversation are involved.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import select
import subprocess
import time


class AppTools:
    def __init__(self, server: Path, thread_id: str):
        self.thread_id = thread_id
        self.sequence = 0
        self.process = subprocess.Popen(
            [str(server.parent/'scripts/launch_codex_app_tools_mcp'), str(server)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self.request('initialize', {'protocolVersion': '2024-11-05',
                     'capabilities': {}, 'clientInfo': {'name': 'getscore-completion-hook', 'version': '1.0'}})
        self.process.stdin.write(json.dumps({'jsonrpc': '2.0', 'method': 'notifications/initialized'})+'\n')
        self.process.stdin.flush()

    def request(self, method, params):
        self.sequence += 1
        request_id = self.sequence
        self.process.stdin.write(json.dumps({'jsonrpc': '2.0', 'id': request_id,
                                            'method': method, 'params': params})+'\n')
        self.process.stdin.flush()
        deadline = time.monotonic()+60
        while time.monotonic() < deadline:
            if not select.select([self.process.stdout], [], [], max(0, deadline-time.monotonic()))[0]:
                break
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError('App tools MCP server closed')
            response = json.loads(line)
            if response.get('id') != request_id:
                continue
            if 'error' in response:
                raise RuntimeError(str(response['error']))
            result = response['result']
            if result.get('isError'):
                raise RuntimeError(str(result))
            return result
        raise TimeoutError('App tools MCP response timed out')

    def call(self, name, arguments):
        return self.request('tools/call', {'name': name, 'arguments': arguments,
                            '_meta': {'openai/threadId': self.thread_id}})

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--thread-id', required=True)
    parser.add_argument('--server', type=Path, required=True)
    parser.add_argument('--completion-prompt-file', type=Path,
                        help='Use this batch-specific completion message')
    parser.add_argument('--probe', action='store_true', help='Read this thread; do not send a message')
    args = parser.parse_args()
    batch = args.batch.resolve()
    if args.probe:
        client = AppTools(args.server, args.thread_id)
        try:
            catalog = client.request('tools/list', {})['tools']
            names = [t['name'] for t in catalog]
            assert 'send_message_to_thread' in names, names
            client.call('read_thread', {'threadId': args.thread_id, 'turnLimit': 1,
                                      'includeOutputs': False, 'maxOutputCharsPerItem': 100})
            (batch/'completion-hook-probe.json').write_text(json.dumps({
                'status': 'passed', 'transport': 'installed codex-app-tools MCP stdio',
                'thread_id': args.thread_id, 'message_sent': False,
                'wakeup_tool_available': True, 'time_unix': time.time()}, indent=2)+'\n')
            print('Completion hook transport verified; no message sent.')
        finally:
            client.close()
        return
    record = {'status': 'waiting', 'supervisor_pid': args.pid, 'thread_id': args.thread_id,
              'mechanism': 'Linux pidfd process-exit event -> codex-app-tools send_message_to_thread',
              'registered_at_unix': time.time()}
    status_path = batch/'completion-hook.json'
    status_path.write_text(json.dumps(record, indent=2)+'\n')
    try:
        try:
            pidfd = os.pidfd_open(args.pid)
        except ProcessLookupError:
            pidfd = None
        if pidfd is not None:
            try:
                select.select([pidfd], [], [])
            finally:
                os.close(pidfd)
        prompt = (
            '后台 long original/swap benchmark 的收尾进程已经退出；这是用户授权的完成事件回调。'
            f'请读取 {batch}/completion-supervisor.json、final-verification.json、report.txt，'
            '核验总数、成功率以及每例视频和目录完整性，然后在本会话向用户报告最终结果。'
            '如果记录显示异常退出或未完成，请如实报告并处理基础设施问题；不要重跑已有有效结果。'
            '用户要求停止持续监控，此回调只在进程结束时触发一次。'
        )
        if args.completion_prompt_file:
            prompt = args.completion_prompt_file.read_text()
        client = AppTools(args.server, args.thread_id)
        try:
            result = client.call('send_message_to_thread', {'threadId': args.thread_id, 'prompt': prompt})
        finally:
            client.close()
        record.update(status='delivered', delivered_at_unix=time.time(), response=result)
    except Exception as exc:
        record.update(status='delivery_failed', error=repr(exc), finished_at_unix=time.time())
        raise
    finally:
        status_path.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
