"""Check SSH every 30 minutes; wake the authorized chat once after recovery."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from check_experiment_server import check
from notify_long_benchmark_completion import AppTools


def save(path, record):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--thread-id', required=True)
    parser.add_argument('--server', type=Path, required=True)
    parser.add_argument('--check-once', action='store_true', help='Check only; never wake the chat')
    args = parser.parse_args()
    batch = args.batch.resolve()
    state_path = batch / 'server-recovery.json'
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    if previous.get('status') == 'delivered' and not args.check_once:
        return
    attempt = 0
    while True:
        started = time.monotonic()
        attempt += 1
        result = check()
        record = dict(status='checking' if args.check_once else 'waiting_for_server',
                      pid=os.getpid(), interval_seconds=1800, attempt=attempt,
                      thread_id=args.thread_id, check=result,
                      next_check_at_unix=time.time() + 1800)
        save(state_path, record)
        if args.check_once:
            print(json.dumps(result, ensure_ascii=False))
            return
        if result['ssh_ready']:
            record.update(status='sending_recovery_event', next_check_at_unix=None)
            save(state_path, record)
            prompt = (
                '用户授权的一次性服务器恢复回调：公司服务器 192.168.50.73 已通过 SSH 连通验证。'
                f'请读取 {batch}/server-recovery.json、selection.json、status.json，继续原定 13 项 swap Kimi 失败实验。'
                '先通过 user@192.168.50.73 检查 /mnt/reacher-fast 是否挂载，以及已有挂载配置和模型文件；'
                '用户授权按需要挂载 reacher-fast 盘，使用服务器现有配置。检查或恢复 π0.5:8008、GAWM:8115、感知:8114 服务，'
                '参考 scripts/remote-pi05-libero.sh、scripts/remote-gawm-start.sh 及已有感知服务配置。'
                '然后启动 scripts/run_swap_kimi_2000.py，batch 使用上述目录；模型必须为 kimi-code/k3.1-pro，'
                '13 项从头重跑、上限 2000 步，原 outputs/long120_1027 保持原样。'
                '旧等待进程及旧完成回调已经停止，正式启动实验时重新注册一次性完成回调；'
                'notify_long_benchmark_completion.py 必须使用 /usr/bin/python3（该解释器支持 os.pidfd_open），'
                '并使用该 batch 的 completion-prompt.txt。'
                '本网络检查器已在连通后停止周期检查；无需持续监控或重复唤醒当前会话，实验完成后再汇报。'
            )
            client = None
            try:
                client = AppTools(args.server, args.thread_id)
                response = client.call('send_message_to_thread',
                                       {'threadId': args.thread_id, 'prompt': prompt})
                record.update(status='delivered', delivered_at_unix=time.time(), response=response)
            except Exception as exc:
                record.update(status='delivery_failed', error=repr(exc))
                raise
            finally:
                if client is not None:
                    client.close()
                save(state_path, record)
            return
        time.sleep(max(0, 1800 - (time.monotonic() - started)))


if __name__ == '__main__':
    main()
